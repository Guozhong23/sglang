from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.mem_cache.unified_cache.components.full_component import FullComponent

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams


class PDDecodeFullComponent(FullComponent):
    """Share Full prefixes without sharing their request-private SWA tails."""

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:
        frontier = req.kv.swa_evicted_seqlen
        if not is_finished:
            # Insertion can deduplicate/replace Full pages. Only publish pages
            # whose SWA mappings have already been cleared (or never existed).
            page_size = self.cache.page_size
            return min(token_ids_len, frontier // page_size * page_size)

        # Clear SWA before Full insertion can free duplicate Full pages. The
        # generic finished path still owns Full/tail frees and the prefix lock.
        if frontier < token_ids_len:
            indices = self.cache.req_to_token_pool.req_to_token[
                req.req_pool_idx, frontier:token_ids_len
            ]
            self.cache.token_to_kv_pool_allocator.free_swa(indices)
        return None
