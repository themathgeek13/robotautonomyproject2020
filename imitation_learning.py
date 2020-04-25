from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
import numpy as np


class ImitationLearning(object):

    def predict_action(self, batch):
        return np.random.uniform(size=(len(batch), 7))

    def behaviour_cloning_loss(self, ground_truth_actions, predicted_actions):
        return 1

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3)
        self.pool = nn.MaxPool2d(2, 2)      # after this, size 64 x 64 x 10
        self.conv2 = nn.Conv2d(10, 25, 3)
        # self.pool = nn.MaxPool2d(2, 2)      # after this, size 32 x 32 x 25
        self.fc1 = nn.Linear(30 * 30 * 25, 16 * 16 * 10)
        self.fc2 = nn.Linear(16 * 16 * 10, 8 * 8 * 5)
        self.fc3 = nn.Linear(8 * 8 * 5, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 30 * 30 * 25)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

import torch.optim as optim
import torch
criterion = nn.MSELoss(reduction = 'sum')
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# To use 'saved' demos, set the path below, and set live_demos=False
live_demos = True
DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'

obs_config = ObservationConfig()
obs_config.set_all(True)

action_mode = ActionMode(ArmActionMode.ABS_EE_POSE)
env = Environment(
    action_mode, DATASET, obs_config, False, static_positions=True)
env.launch()

task = env.get_task(EmptyContainer)

il = ImitationLearning()

for i in range(10):
    print("Demo set: ", i)
    demos = task.get_demos(10, live_demos=live_demos)  # -> List[List[Observation]]
    demos = np.array(demos).flatten()
    np.savez("demo"+str(i)+".npy", demos)
print('Done')
env.shutdown()

# An example of using the demos to 'train' using behaviour cloning loss.
# for i in range(100):
#     print("'training' iteration %d" % i)
#     batch = np.random.choice(demos, replace=False)
#     batch_images = [obs.left_shoulder_rgb for obs in batch]
#     predicted_actions = il.predict_action(batch_images)
#     ground_truth_actions = [obs.joint_velocities for obs in batch]
#     loss = il.behaviour_cloning_loss(ground_truth_actions, predicted_actions)
inputs, labels = [], []

for d in demos:
    for obs in d:
        inputs.append(obs.wrist_rgb)
        labels.append(obs.gripper_pose)
inputs = torch.from_numpy(np.array(inputs))
labels = torch.from_numpy(np.array(labels))

for epoch in range(2):
    running_loss = 0.0
    for i in range(len(inputs)):
        inputdata = inputs[i]
        labeldata = torch.unsqueeze(labels[i].float(),0)
        inputdata = torch.unsqueeze(inputdata.permute(2,0,1), 0)
        optimizer.zero_grad()
        outputs = net(inputdata)
        loss = criterion(outputs, labeldata)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i%20==19:
            print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0
