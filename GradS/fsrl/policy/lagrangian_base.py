from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.utils import MultipleLRSchedulers
from torch import nn
from torch.nn.functional import relu
from scipy.special import softmax

from fsrl.policy import BasePolicy
from fsrl.utils import BaseLogger
from fsrl.utils.optim_util import LagrangianOptimizer



class LagrangianPolicy(BasePolicy):
    """Implementation of PID Lagrangian-based method.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~fsrl.policy.BasePolicy`. (s -> logits)
    :param List[torch.nn.Module] critics: a list of the critic network. (s -> V(s))
    :param dist_fn: distribution class for computing the action. :type dist_fn:
        Type[torch.distributions.Distribution]
    :param BaseLogger logger: dummy logger for logging events.
    :param bool use_lagrangian: whether to use Lagrangian method. Default to True.
    :param list lagrangian_pid: list of PID constants for Lagrangian multiplier. Default
        to [0.05, 0.0005, 0.1].
    :param float cost_limit: cost limit for the Lagrangian method. Default to np.inf.
    :param bool rescaling: whether use the rescaling trick for Lagrangian multiplier, see
        Alg. 1 in http://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf
    :param float gamma: the discounting factor for cost and reward, should be in [0, 1].
        Default to 0.99.
    :param int max_batchsize: the maximum size of the batch when computing GAE, depends
        on the size of available memory and the memory cost of the model; should be as
        large as possible within the memory constraint. Default to 99999.
    :param bool reward_normalization: normalize estimated values to have std close to 1.
        Default to False.
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to True.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1]. can be
        either “clip” (for simply clipping the action) or empty string for no bounding.
        Default to "clip".
    :param gym.Space observation_space: environment's observation space. Default to None.
    :param gym.Space action_space: environment's action space. Default to None.
    :param lr_scheduler: learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None.

    .. seealso::

        Please refer to :class:`~fsrl.policy.BasePolicy` for more detailed explanation.
    """

    def __init__(
        self,
        actor: nn.Module,
        critics: Union[nn.Module, List[nn.Module]],
        dist_fn: Type[torch.distributions.Distribution],
        logger: BaseLogger = BaseLogger(),
        # Lagrangian specific arguments
        use_lagrangian: bool = True,
        lagrangian_pid: Tuple = (0.05, 0.0005, 0.1),
        cost_limit: Union[List, float] = np.inf,
        rescaling: bool = True,
        # Based policy common arguments
        gamma: float = 0.99,
        max_batchsize: int = 99999,
        reward_normalization: bool = False,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        lr_scheduler: Optional[Union[torch.optim.lr_scheduler.LambdaLR,
                                     MultipleLRSchedulers]] = None,
        vanilla: bool = True,
        random_con_selection: bool = False,
        gradient_shaping: bool = False,
        max_singleAC: bool = False,
        optim=None

    ) -> None:
        super().__init__(
            actor, critics, dist_fn, logger, gamma, max_batchsize, reward_normalization,
            deterministic_eval, action_scaling, action_bound_method, observation_space,
            action_space, lr_scheduler
        )
        self.vanilla = vanilla
        self.random_con_selection = random_con_selection
        self.gradient_shaping = gradient_shaping
        self.max_singleAC = max_singleAC
        self.optim = optim

        self.rescaling = rescaling
        self.use_lagrangian = use_lagrangian
        self.cost_limit = [cost_limit] * (self.critics_num -
                                          1) if np.isscalar(cost_limit) else cost_limit
        # suppose there are M constraints, then critics_num = M + 1
        if self.use_lagrangian:
            assert len(
                self.cost_limit
            ) == (self.critics_num - 1), "cost_limit must has equal len of critics_num"
            self.lag_optims = [
                LagrangianOptimizer(lagrangian_pid) for _ in range(self.critics_num - 1)
            ]
        else:
            self.lag_optims = []
        self.sim_threshold = 0.8
        self.conflicting_threshold = 0.999
        self.activated_index = -1 # for TRPO only

    def pre_update_fn(self, stats_train: Dict, **kwarg) -> None:
        cost_values = stats_train["m_cost"]  # TODO: modified in the collector / env. to extend to 3 or higher dimension
        self.update_lagrangian(cost_values)

    def update_cost_limit(self, cost_limit: float) -> None:
        """Update the cost limit threshold.

        :param float cost_limit: new cost threshold
        """
        self.cost_limit = [cost_limit] * (self.critics_num -
                                          1) if np.isscalar(cost_limit) else cost_limit

    def update_lagrangian(self, cost_values: Union[List, float]) -> None:
        """Update the Lagrangian multiplier before updating the policy.

        :param Union[List, float] cost_values: the estimation of cost values that want to
            be controlled under the target thresholds. It could be a list (multiple
            constraints) or a scalar value.
        """
        if np.isscalar(cost_values):
            cost_values = [cost_values]
        for i, lag_optim in enumerate(self.lag_optims):
            lag_optim.step(cost_values[i], self.cost_limit[i])

    def get_extra_state(self):
        """Save the lagrangian optimizer's parameters.

        This function is called when call the policy.state_dict(), see
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_extra_state
        """
        if len(self.lag_optims):
            return [optim.state_dict() for optim in self.lag_optims]
        else:
            return None

    def set_extra_state(self, state):
        """Load the lagrangian optimizer's parameters.

        This function is called from load_state_dict() to handle any extra state found
        within the state_dict.
        """
        if "_extra_state" in state:
            lag_optim_cfg = state["_extra_state"]
            if lag_optim_cfg and self.lag_optims:
                for i, state_dict in enumerate(lag_optim_cfg):
                    self.lag_optims[i].load_state_dict(state_dict)

    # def safety_loss(self, values: List) -> Tuple[torch.tensor, dict]:
    #     """Compute the safety loss based on Lagrangian and return the scaling factor.
    #     :param list values: the cost values that want to be constrained. They will be
    #         multiplied with the Lagrangian multipliers.
    #     :return tuple[torch.tensor, dict]: the total safety loss and a dictionary of info
    #         (including the rescaling factor, lagrangian, safety loss etc.)
    #     """

    #     # Modified to calculate the gradient similarity

    #     # get a list of lagrangian multiplier
    #     lags = [optim.get_lag() for optim in self.lag_optims]
    #     # Alg. 1 of http://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf
    #     rescaling = 1. / (np.sum(lags) + 1) if self.rescaling else 1
    #     assert len(values) == len(lags), "lags and values length must be equal"
    #     stats = {"loss/rescaling": rescaling}
    #     loss_safety_total = 0.
    #     safety_loss_list = []
    #     for i, (value, lagrangian) in enumerate(zip(values, lags)):
    #         safety_loss_i = torch.mean(value)
    #         safety_loss_list.append(safety_loss_i)

    #         loss = torch.mean(value * lagrangian)
    #         loss_safety_total += loss
    #         suffix = "" if i == 0 else "_" + str(i)
    #         stats["loss/lagrangian" + suffix] = lagrangian
    #         stats["loss/actor_safety" + suffix] = loss.item()

    #     self.optim.zero_grad()
    #     safety_loss_grad_list = []
    #     for i in range(len(safety_loss_list)):

    #         self.optim.zero_grad()
    #         safety_loss_list[i].backward(retain_graph=True)

    #         grads = []
    #         for param in self.actor.parameters():
    #             grads.append(param.grad.view(-1))
    #         grads = torch.cat(grads)
    #         safety_loss_grad_list.append(grads)
    #         self.optim.zero_grad()

    #     cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    #     # Storing the gradient simalirity
    #     for i in range(1, len(safety_loss_grad_list)):
    #         sim = torch.mean(cos(safety_loss_grad_list[0], safety_loss_grad_list[i])).item()
    #         stats["grad_sim/0grad_"+str(i)] = sim

    #     for i in range( len(safety_loss_grad_list) - 1):
    #         sim = torch.mean(cos(safety_loss_grad_list[-1], safety_loss_grad_list[i])).item()
    #         stats["grad_sim/-1grad_"+str(i)] = sim

    #     return loss_safety_total, stats

    def safety_loss(self, values: List, required_GS = True) -> Tuple[torch.tensor, dict]:
        """Compute the safety loss based on Lagrangian and return the scaling factor.

        :param list values: the cost values that want to be constrained. They will be
            multiplied with the Lagrangian multipliers.
        :return tuple[torch.tensor, dict]: the total safety loss and a dictionary of info
            (including the rescaling factor, lagrangian, safety loss etc.)
        """
        # get a list of lagrangian multiplier
        lags = [optim.get_lag() for optim in self.lag_optims]
        # Alg. 1 of http://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf
        safety_loss_grad_list = []
        if self.optim is not None and required_GS:
            stats = {"loss/cal_sim": 1}
            vanilla_loss_safety_total = 0
            safety_loss_list = []
            for i, (value, lagrangian) in enumerate(zip(values, lags)):
                safety_loss_i = torch.mean(value)
                safety_loss_list.append(safety_loss_i)

                loss = torch.mean(value * lagrangian)
                vanilla_loss_safety_total += loss
                suffix = "" if i == 0 else "_" + str(i)
                stats["loss/lagrangian" + suffix] = lagrangian
                stats["loss/actor_safety" + suffix] = loss.item()

            self.optim.zero_grad()
            safety_loss_grad_list = []
            for i in range(len(safety_loss_list)):
                # print(i)
                self.optim.zero_grad()
                safety_loss_list[i].backward(retain_graph=True)

                grads = []
                for param in self.actor.parameters():
                    grads.append(param.grad.view(-1))
                grads = torch.cat(grads)
                safety_loss_grad_list.append(grads)
                self.optim.zero_grad()

            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            # Storing the gradient simalirity
            for i in range(1, len(safety_loss_grad_list)):
                sim = torch.mean(cos(safety_loss_grad_list[0], safety_loss_grad_list[i])).item()
                stats["grad_sim/0grad_"+str(i)] = sim

            for i in range( len(safety_loss_grad_list) - 1):
                sim = torch.mean(cos(safety_loss_grad_list[-1], safety_loss_grad_list[i])).item()
                stats["grad_sim/-1grad_"+str(i)] = sim

        else:
            stats = {"loss/cal_sim": 0}

        rescaling = 1. / (np.sum(lags) + 1) if self.rescaling else 1
        stats = {"loss/rescaling": rescaling}
        loss_safety_total = 0.
        safety_loss_list = []
        for i, (value, lagrangian) in enumerate(zip(values, lags)):
            safety_loss_i = torch.mean(value)
            safety_loss_list.append(safety_loss_i)

            loss = torch.mean(value * lagrangian)
            loss_safety_total += loss
            suffix = "" if i == 0 else "_" + str(i)
            stats["loss/lagrangian" + suffix] = lagrangian
            stats["loss/actor_safety" + suffix] = loss.item()

        if self.vanilla:
            pass

        else:
            # with torch.no_grad():
            norm_lag = np.array(lags)
            for i in range(len(norm_lag)):
                norm_lag[i] = norm_lag[i] / self.cost_limit[i]

            if self.random_con_selection:
                max_index = np.random.randint(len(lags)) # % 
            elif self.max_singleAC:
                max_index = np.argmax(np.array(lags))
            elif self.gradient_shaping and required_GS:
                if len(safety_loss_grad_list) < 1:
                    lam_dis = softmax(np.array(norm_lag))
                    max_index = np.random.choice(range(0, len(lags)), p=lam_dis)
                else:
                    index_list = [0]
                    for index_i in range(1, len(lags)):
                        sim_list = []

                        for index_j in index_list:
                            sim_ij = torch.mean(cos(safety_loss_grad_list[index_i], safety_loss_grad_list[index_j])).item()
                        sim_list.append(sim_ij)
                        
                        if np.max(np.array(sim_list)) < self.sim_threshold and np.min(np.array(sim_list)) > -1 * self.conflicting_threshold:
                            index_list.append(index_i)
                    selected_norm_lag = []
                    
                    for i in range(len(index_list)):
                        norm_lag_index = index_list[i]
                        selected_norm_lag_i = norm_lag[norm_lag_index]
                        selected_norm_lag.append(selected_norm_lag_i)

                    lam_dis = softmax(np.array(selected_norm_lag))
                    # max_index_selected = np.random.randint(len(index_list))
                    max_index_selected = np.random.choice(range(0, len(index_list)), p=lam_dis)
                    max_index = index_list[max_index_selected]
                    # print("len(index_list)", len(index_list))
                    # print("max_index", max_index)
            elif self.gradient_shaping and not required_GS: # for TRPO line searching only:
                assert self.activated_index >= 0
                max_index = self.activated_index
                
            else:
                raise NotImplementedError
            
            self.activated_index = max_index
            
            rescaling = 1. / (lags[max_index] + 1) if self.rescaling else 1 # 
            stats["loss/rescaling"] = rescaling
            loss_safety_total = 0.
            for i, (value, lagrangian) in enumerate(zip(values, lags)):
                loss = torch.mean(value * lagrangian)
                if i == max_index:
                    loss_safety_total += loss
                suffix = "_" + str(i)
                stats["loss/lagrangian" + suffix] = lagrangian
                stats["loss/actor_safety" + suffix] = loss.item()

        return loss_safety_total, stats

class LagrangianCDPolicy(LagrangianPolicy):
    """
        Updating the lagrangian multiplier using cordinate descent
    """
    def __init__(
        self,
        actor: nn.Module,
        critics: Union[nn.Module, List[nn.Module]],
        dist_fn: Type[torch.distributions.Distribution],
        logger: BaseLogger = BaseLogger(),
        # Lagrangian specific arguments
        use_lagrangian: bool = True,
        lagrangian_pid: Tuple = (0.05, 0.0005, 0.1),
        cost_limit: Union[List, float] = np.inf,
        rescaling: bool = True,
        # Based policy common arguments
        gamma: float = 0.99,
        max_batchsize: int = 99999,
        reward_normalization: bool = False,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        lr_scheduler: Optional[Union[torch.optim.lr_scheduler.LambdaLR,
                                     MultipleLRSchedulers]] = None
    ) -> None:
        super().__init__(
            actor=actor,
            critics=critics,
            dist_fn=dist_fn,
            logger=logger,
            use_lagrangian=use_lagrangian,
            lagrangian_pid=lagrangian_pid,
            cost_limit=cost_limit,
            rescaling=rescaling,
            gamma=gamma,
            max_batchsize=max_batchsize,
            reward_normalization=reward_normalization,
            deterministic_eval=deterministic_eval,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            observation_space=observation_space,
            action_space=action_space,
            lr_scheduler=lr_scheduler
        )

    def update_lagrangian(self, cost_values: Union[List, float]) -> None:
        """using coordinate descent"""

        if np.isscalar(cost_values):
            cost_values = [cost_values]

        constraint_list = []
        for i in range(len(self.cost_limit)):
            g = cost_values[i]-self.cost_limit[i]
            constraint_list.append(g)
        
        constraint_list = np.array(constraint_list)
        index = np.argmax(constraint_list)


        for i, lag_optim in enumerate(self.lag_optims):
            if i == index:
                lag_optim.step(cost_values[i], self.cost_limit[i])


class LagrangianUPolicy(LagrangianPolicy):
    """
        All the constraints share the same (uniform) lagrangian multiplier
    """
    def __init__(
        self,
        actor: nn.Module,
        critics: Union[nn.Module, List[nn.Module]],
        dist_fn: Type[torch.distributions.Distribution],
        logger: BaseLogger = BaseLogger(),
        # Lagrangian specific arguments
        use_lagrangian: bool = True,
        lagrangian_pid: Tuple = (0.05, 0.0005, 0.1),
        cost_limit: Union[List, float] = np.inf,
        rescaling: bool = True,
        # Based policy common arguments
        gamma: float = 0.99,
        max_batchsize: int = 99999,
        reward_normalization: bool = False,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        lr_scheduler: Optional[Union[torch.optim.lr_scheduler.LambdaLR,
                                     MultipleLRSchedulers]] = None,
        random_selection: bool = False
    ) -> None:
        super().__init__(
            actor=actor,
            critics=critics,
            dist_fn=dist_fn,
            logger=logger,
            use_lagrangian=use_lagrangian,
            lagrangian_pid=lagrangian_pid,
            cost_limit=cost_limit,
            rescaling=rescaling,
            gamma=gamma,
            max_batchsize=max_batchsize,
            reward_normalization=reward_normalization,
            deterministic_eval=deterministic_eval,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            observation_space=observation_space,
            action_space=action_space,
            lr_scheduler=lr_scheduler
        )
        self.random_selection = random_selection

    def update_lagrangian(self, cost_values: Union[List, float]) -> None:

        if np.isscalar(cost_values):
            cost_values = [cost_values]

        constraint_list = []
        info = {}
        for i in range(len(self.cost_limit)):
            g = cost_values[i]-self.cost_limit[i]
            constraint_list.append(g)
            info["loss/constraint_"+str(i)] = g
        
        constraint_list = np.array(constraint_list)
        index = np.argmax(constraint_list)
        info["loss/constraint_index"] = index

        self.logger.store(**info)

        for i, lag_optim in enumerate(self.lag_optims):
            lag_optim.step(cost_values[index], self.cost_limit[index])
            return # Only update for once.

        
    def safety_loss(self, values: List) -> Tuple[torch.tensor, dict]:
        """Compute the safety loss based on Lagrangian and return the scaling factor.

        :param list values: the cost values that want to be constrained. They will be
            multiplied with the Lagrangian multipliers.
        :return tuple[torch.tensor, dict]: the total safety loss and a dictionary of info
            (including the rescaling factor, lagrangian, safety loss etc.)
        """
        # get a list of lagrangian multiplier
        lags = [optim.get_lag() for optim in self.lag_optims]
        # Alg. 1 of http://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf
        rescaling = 1. / (np.sum(lags) + 1) if self.rescaling else 1

        constraint_list = []
        for i in range(len(values)):
            g = (values[i]-self.cost_limit[i]).mean().detach().cpu().numpy()
            constraint_list.append(g)
        constraint_list = np.array(constraint_list)
        index = np.argmax(constraint_list)
        value_max = values[index]

        assert len(values) == len(lags), "lags and values length must be equal"
        stats = {"loss/rescaling": rescaling}
        loss_safety_total = 0.
        for i, (value, lagrangian) in enumerate(zip(values, lags)):
            loss = torch.mean(value_max * lagrangian)
            loss_safety_total += loss
            suffix = "" if i == 0 else "_" + str(i)
            stats["loss/lagrangian" + suffix] = lagrangian
            stats["loss/actor_safety" + suffix] = loss.item()
            break
        return loss_safety_total, stats


class LagrangianUPolicy_modified(LagrangianPolicy):
    """
        All the constraints share the same (uniform) lagrangian multiplier
    """
    def __init__(
        self,
        actor: nn.Module,
        critics: Union[nn.Module, List[nn.Module]],
        dist_fn: Type[torch.distributions.Distribution],
        logger: BaseLogger = BaseLogger(),
        # Lagrangian specific arguments
        use_lagrangian: bool = True,
        lagrangian_pid: Tuple = (0.05, 0.0005, 0.1),
        cost_limit: Union[List, float] = np.inf,
        rescaling: bool = True,
        # Based policy common arguments
        gamma: float = 0.99,
        max_batchsize: int = 99999,
        reward_normalization: bool = False,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        lr_scheduler: Optional[Union[torch.optim.lr_scheduler.LambdaLR,
                                     MultipleLRSchedulers]] = None,
        random_con_selection: bool = False
    ) -> None:
        super().__init__(
            actor=actor,
            critics=critics,
            dist_fn=dist_fn,
            logger=logger,
            use_lagrangian=use_lagrangian,
            lagrangian_pid=lagrangian_pid,
            cost_limit=cost_limit,
            rescaling=rescaling,
            gamma=gamma,
            max_batchsize=max_batchsize,
            reward_normalization=reward_normalization,
            deterministic_eval=deterministic_eval,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            observation_space=observation_space,
            action_space=action_space,
            lr_scheduler=lr_scheduler
        )
        self.random_con_selection = random_con_selection

        
    def safety_loss(self, values: List) -> Tuple[torch.tensor, dict]:
        """Compute the safety loss based on Lagrangian and return the scaling factor.

        :param list values: the cost values that want to be constrained. They will be
            multiplied with the Lagrangian multipliers.
        :return tuple[torch.tensor, dict]: the total safety loss and a dictionary of info
            (including the rescaling factor, lagrangian, safety loss etc.)
        """
        # get a list of lagrangian multiplier
        lags = [optim.get_lag() for optim in self.lag_optims]
        if not self.random_con_selection:
            max_index = np.argmax(np.array(lags)) # %
        else:
            max_index = np.random.randint(len(lags))
        # Alg. 1 of http://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf
        rescaling = 1. / (lags[max_index] + 1) if self.rescaling else 1 # %

        # constraint_list = []
        # constraint_info = {}
        # for i in range(len(values)):
        #     g = (values[i]-self.cost_limit[i]).mean().detach().cpu().numpy()
        #     constraint_list.append(g)
        #     constraint_info["g_"+str(i)] = g
        # constraint_list = np.array(constraint_list)
        # index = np.argmax(constraint_list)

        assert len(values) == len(lags), "lags and values length must be equal"
        stats = {
            "loss/rescaling": rescaling,
            "loss/constrain_index": max_index
        }

        constrain_index = {
            "constraint/constrain_index": max_index
        }

        self.logger.store(**constrain_index)
        # self.logger.store(**constraint_info, tab="constraint")

        loss_safety_total = 0.
        for i, (value, lagrangian) in enumerate(zip(values, lags)):
            loss = torch.mean(value * lagrangian)
            if i == max_index:
                # loss = torch.mean(value * lagrangian)
                loss_safety_total += loss
            suffix = "" if i == 0 else "_" + str(i)
            stats["loss/lagrangian" + suffix] = lagrangian
            stats["loss/actor_safety" + suffix] = loss.item()
        return loss_safety_total, stats