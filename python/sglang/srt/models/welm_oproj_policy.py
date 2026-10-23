"""Host-only, immutable OProj dispatch policy for a WeLM runner."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WelmOProjPolicy:
    use_mc2: bool = False
    max_pipeline_chunks: int = 0
    min_chunk_tokens: int = 1024

    def __post_init__(self):
        if self.max_pipeline_chunks < 0 or self.min_chunk_tokens < 1:
            raise ValueError(
                "WeLM OProj requires nonnegative chunks and positive tokens"
            )

    def select(self, *, rows, tp_size, eligible, is_dp_graph):
        """Return (pipeline chunks, MC2); never switch after submitting work."""
        if is_dp_graph or not eligible or tp_size < 2 or rows <= 0:
            return 0, False
        if rows % tp_size:
            return 0, False
        min_local_rows = (self.min_chunk_tokens + tp_size - 1) // tp_size
        chunks = min(self.max_pipeline_chunks, (rows // tp_size) // min_local_rows)
        if chunks >= 2:
            return chunks, False
        return 0, self.use_mc2
