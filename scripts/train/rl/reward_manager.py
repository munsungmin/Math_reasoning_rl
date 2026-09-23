"""Native reward execution plus data needed by common prediction metrics."""

from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager


class RecordedRewardManager(NaiveRewardManager):
    async def run_single(self, data):
        result = await super().run_single(data)
        item = data[-1]
        response = item.batch["responses"]
        length = int(item.batch["attention_mask"][-response.shape[-1] :].sum())
        info = result["reward_extra_info"]
        info.update(
            sample_id=str(item.non_tensor_batch["extra_info"]["sample_id"]),
            dataset=str(item.non_tensor_batch["data_source"]),
            generated_tokens=length,
            hit_length_limit=bool(
                length and int(response[length - 1]) != self.tokenizer.eos_token_id
            ),
        )
        return result
