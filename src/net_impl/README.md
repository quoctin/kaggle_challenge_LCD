# How to use network.py?
- Change hyper parameters.
- Change predefined constants.
- Run the code.

# How to load data in batch?
In order to load data in batches, call function *next_training_batch(batch_size=5, batch_offset=0, slices=141)*.

* **batch_size**: the size of batch you wanna load.
* **batch_offset**: this is the current position of the next batch. This parameter is saved in terms of *global_batch_offset* in order to let the training resume properly after interruption.
* **slices**: the number of slices / patient.

### Examples:
1. Load 1 batch of size 10 from the beginning of the training set.
```python
[data, labels, _] = next_training_batch(batch_size=5, batch_offset=0)
```
2. Load the 5 batches of size 20 from the 10-th sample.
```python
batch_offset = 10
for i in range(5):
    [data, labels, batch_offset] = next_training_batch(batch_size=20, batch_offset=batch_offset)
    # doing something
```