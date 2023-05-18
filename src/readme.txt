# Incremental Learning with Human Pose

## Results:
Base dataset = CIFAR10
Incremental dataset = MNIST

### "Base":
Test with base classes:70.28%
Test with incr. classes:3.16%

### "Freeze":
Test with base classes:7.42%
Test with incr. classes:97.17%

### "AddRegularization":
Test with base classes:15.44%
Test with incr. classes:93.2%

### "LearningWithoutForgetting":
Test with base classes:23.31%
Test with incr. classes:94.85%
