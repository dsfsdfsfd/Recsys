from recsys.config import CustomDatasetSize

class DatasetSampler:
    _SIZES = {
        CustomDatasetSize.LARGE: 50_000,
        CustomDatasetSize.MEDIUM: 5_000,
        CustomDatasetSize.SMALL: 1_000,
    }

    def __init__(self, size: CustomDatasetSize) -> None:
        self._size = size

    @classmethod
    def get_supported_sizes(cls) -> dict:
        return cls._SIZES