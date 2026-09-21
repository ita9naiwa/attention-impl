import torch

from flash_attn.cute.vc_vsa_preprocess import prepare_vsa_routes


def reference(selected, sizes, block, prefix, start):
    capacity = (selected.shape[-1] + prefix) * (block // 128)
    full = torch.full((*selected.shape[:-1], capacity), -1, dtype=torch.int32)
    partial = torch.full_like(full, -1)
    full_count = torch.zeros(selected.shape[:-1], dtype=torch.int32)
    partial_count = torch.zeros_like(full_count)
    for row, ids in enumerate(selected.cpu().reshape(-1, selected.shape[-1]).tolist()):
        parents = sorted(list(range(prefix)) + [i - start for i in ids])
        for output, count, is_full in ((full, full_count, True), (partial, partial_count, False)):
            expanded = []
            for parent in parents:
                size = int(sizes[parent])
                selected_kind = size == block if is_full else 0 < size < block
                if selected_kind:
                    expanded.extend(parent * (block // 128) + child for child in range(block // 128))
            output.reshape(-1, capacity)[row, :len(expanded)] = torch.tensor(expanded, dtype=torch.int32)
            count.reshape(-1)[row] = len(expanded)
    return full, full_count, partial, partial_count


@torch.no_grad()
def test_route_dispatch_boundaries():
    torch.manual_seed(20260921)
    for block in (128, 256):
        for total in (1, 30, 32, 33, 128, 1024, 1025):
            prefix = min(2, total - 1)
            parents, start = total + 7, 11
            selected = torch.stack([
                torch.randperm(parents - prefix)[:total - prefix] + prefix + start
                for _ in range(3)
            ]).reshape(1, 1, 3, total - prefix).cuda()
            sizes = torch.tensor(([0, 1, block - 1, block] * parents)[:parents], dtype=torch.int32, device="cuda")
            for _ in range(3):
                prepare_vsa_routes(selected, sizes, block, prefix, start)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = prepare_vsa_routes(selected, sizes, block, prefix, start)
            for _ in range(3):
                selected.copy_(selected.flip(-1))
                sizes.copy_(sizes.roll(1))
                for output in outputs:
                    output.fill_(-777)
                graph.replay()
                expected = reference(selected, sizes.cpu(), block, prefix, start)
                for actual, wanted in zip(outputs, expected):
                    assert torch.equal(actual.cpu(), wanted), (block, total)


if __name__ == "__main__":
    test_route_dispatch_boundaries()
    print("14 dispatch boundaries / 42 changed Graph replays passed")
