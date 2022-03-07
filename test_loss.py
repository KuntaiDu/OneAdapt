import torch
import math
batch_size = 1
num_classes = 11

loss_fn = torch.nn.BCELoss()

# outputs_before_sigmoid = torch.randn(batch_size, num_classes)
# target_classes = torch.randint(0, 2, (batch_size, num_classes))  # randints in [0, 2).

# sigmoid_outputs = torch.sigmoid(outputs_before_sigmoid)
# sigmoid_outputs = torch.Tensor([[0.3, 0.4, 0.1, 0.88, 0.24, 0.78, 0.9, 0.05, 0.81]])
# target_classes = torch.Tensor([[0, 0, 0, 1, 0, 1, 1, 0, 0]])
sigmoid_outputs = torch.Tensor([[0.9]])
target_classes = torch.Tensor([[0]])
target_classes = target_classes.to(torch.float)
loss_fn_2 = torch.nn.BCEWithLogitsLoss()
# loss2 = loss_fn_2(outputs_before_sigmoid, target_classes)
print(target_classes)
loss = loss_fn(sigmoid_outputs, target_classes)


print(loss)

loss_hand = 0
for i in range(len(sigmoid_outputs[0])):
    loss_hand +=  ((target_classes[0][i]) * math.log(sigmoid_outputs[0][i]) + (1-target_classes[0][i]) * math.log(1-sigmoid_outputs[0][i]))


print(loss_hand/len(sigmoid_outputs[0]))