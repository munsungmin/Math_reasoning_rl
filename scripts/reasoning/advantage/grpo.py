"""Explicit adapter to the pinned native estimator; no second GRPO implementation."""


class GRPOAdvantage:
    def __init__(self, normalize_std=True):
        self.normalize_std = bool(normalize_std)

    def native(self):
        return {"adv_estimator": "grpo", "norm_adv_by_std_in_grpo": self.normalize_std}

    def __call__(self, rewards, response_mask, group_ids):
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

        return compute_grpo_outcome_advantage(
            rewards,
            response_mask,
            group_ids,
            norm_adv_by_std_in_grpo=self.normalize_std,
        )
