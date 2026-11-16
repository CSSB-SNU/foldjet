"""Helper: list of (position, index, spares) keys for a rank/start, using the adapter's own sampler code (adapter dir given by argv/ENV FT_ADAPTER_DIR)."""

import sys


def get_keys(adapter_dir, rank, world, start, n, seed=0, epoch_size=1 << 17):
    sys.path.insert(0, adapter_dir)
    import ft_adapter
    import mwconfig
    import mwsample
    from miniworld.data.dataloader.sampler import WeightedSampler

    s = ft_adapter.Settings()
    data = mwconfig.load_data_dict(s.data_config, s.overlay)
    full = mwsample.build_dataset(data)
    mk = lambda ns, sd: WeightedSampler(
        full,
        full.weights,
        num_samples=ns,
        num_replicas=world,
        rank=rank,
        shuffle=True,
        seed=sd,
        replacement=full.config.DB_config.sample_with_replacement,
    )
    sampler = mk(epoch_size, seed)
    spare = mk(epoch_size * 16, seed + 1_000_003)
    assert start + n <= epoch_size
    sampler.set_epoch(0)
    spare.set_epoch(0)
    idx = list(iter(sampler))[start : start + n]
    import numpy as np

    sp = np.fromiter(iter(spare), dtype=np.int64, count=epoch_size * 16).reshape(
        epoch_size, 16
    )[start : start + n]
    return [(start + i, int(idx[i]), sp[i].tolist()) for i in range(n)]
