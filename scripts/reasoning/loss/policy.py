"""Compose policy/KL/entropy/reduction settings consumed by native verl."""


class PolicyObjective:
    def __init__(
        self,
        policy="vanilla",
        clip_low=0.2,
        clip_high=0.2,
        dual_clip=3.0,
        kl_coefficient=0.01,
        kl_type="low_var_kl",
        entropy_coefficient=0.0,
        reduction="token-mean",
    ):
        self.policy = policy
        self.clip_low, self.clip_high, self.dual_clip = clip_low, clip_high, dual_clip
        self.kl_coefficient, self.kl_type = kl_coefficient, kl_type
        self.entropy_coefficient, self.reduction = entropy_coefficient, reduction

    def native(self):
        return {
            "policy_loss": {"loss_mode": self.policy},
            "clip_ratio": self.clip_low,
            "clip_ratio_low": self.clip_low,
            "clip_ratio_high": self.clip_high,
            "clip_ratio_c": self.dual_clip,
            "use_kl_loss": self.kl_coefficient != 0,
            "kl_loss_coef": self.kl_coefficient,
            "kl_loss_type": self.kl_type,
            "entropy_coeff": self.entropy_coefficient,
            "calculate_entropy": self.entropy_coefficient != 0,
            "loss_agg_mode": self.reduction,
        }

    def policy_loss(self, old_log_prob, log_prob, advantages, response_mask):
        from omegaconf import OmegaConf
        from verl.trainer.ppo.core_algos import get_policy_loss_fn

        cfg = OmegaConf.create({**self.native(), "global_batch_info": {}})
        return get_policy_loss_fn(self.policy)(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            config=cfg,
            loss_agg_mode=self.reduction,
        )
