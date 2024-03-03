import gcsfs
from datasets import load_dataset, Dataset, IterableDataset


def iter_gcs_files(dirname):
    fs = gcsfs.GCSFileSystem(project='gpt-tpu-370700')
    files = fs.ls(dirname)
    files = [f"gs://{f}" for f in files if f.endswith('.arrow')]
    print(f"{len(files)} files found")

    for file in files:
        with fs.open(file) as f:
            ds = Dataset.from_buffer(f.read())
            yield from ds

it = IterableDataset.from_generator(
    iter_gcs_files,
    gen_kwargs={"dirname": "gs://heegyu-kogpt/data-plm/ko-tiny-llama/train/"}
    )

for i, item in enumerate(it):
    print(i, item)
    if i == 10:
        break

# fs = gcsfs.GCSFileSystem(project='gpt-tpu-370700')
# path = 'gs://heegyu-kogpt/data-plm/ko-tiny-llama/train/data-00000-of-00200.arrow'
# with fs.open(path) as f:
#     ds = Dataset.from_buffer(f.read())
# print(ds)
# print(ds[:4])

# dirname = "heegyu-kogpt/data-plm/ko-tiny-llama/train"
# files = fs.ls(dirname)
# print(files)

# files = [f"gs://{f}" for f in files if f.endswith('.arrow')]

# storage_options = {"project": "gpt-tpu-370700"}
# dataset = load_dataset('parquet', data_files={'train': files}, split='train', streaming=True, storage_options=storage_options)
# print(next(iter(dataset)))