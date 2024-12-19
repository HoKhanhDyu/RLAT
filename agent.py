from arch import *
from config import *
from dataset import *
from torch.utils.data import Dataset, DataLoader
import random
from torchvision.utils import save_image
import os
from collections import deque
import torch.optim as optim
import copy
from tqdm import tqdm
import json
import torch.nn.functional as F
from utils import save_img_mnist

class Agent_Attack():
    def __init__(self, load=False, device="cpu"):
        torch.cuda.current_device()
        self.EPS = EPS
        self.epsilon = EPSILON
        self.device = device
        classifier = MNIST_CC().eval()
        classifier.load_state_dict(torch.load(r"trained_model/mnist_cc.pth"))
        self.classifier = classifier.cuda()
        
        self.img_size = IMG_SIZE
        self.max_iter = MAX_ITER

        self.input_size = INPUT_DQN_SIZE
        self.output_size = OUTPUT_DQN_SIZE
        
        self.policy_net =  DQN(self.input_size, self.output_size).cuda().train()
        
        self.memory = deque(maxlen=300000)
        self.success_memory = deque(maxlen=300)
        self.history = deque([0] * 32, maxlen=32)
        self.criterion = nn.MSELoss().cuda()
        self.TARGET_UPDATE =TARGET_UPDATE
        self.GAMMA = GAMMA

        self.num = 0
        
        if load:
            print("Loading Trained model")
            self.policy_net.load_state_dict(torch.load(r"/kaggle/input/model-trien-vong-2/model_0_trrenvong_2.pth"))

        self.target_net = DQN(self.input_size, self.output_size).cuda()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=1e-4)
        train_dataset = get_dataset('mnist', split='train')
        self.train_dataset = train_dataset
        test_dataset = get_dataset('mnist', split='test')
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        self.action_lists = []
        self.reward_lists = []
        self.loss_lists = []

    def reward(self, img, prob, orig_img, label):
        new_prob = self.classifier_img(img)
        img_diff = torch.norm(img - orig_img) 
        prob_diff = torch.norm(prob - new_prob) 
        new_label = new_prob.argmax()
        if new_label != label[0]:
            return torch.tensor(10).float()
        return prob_diff - 2*img_diff

    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_size - 1)
        with torch.no_grad():
            return self.policy_net(state.cuda()).cpu().argmax().item()
        
    def optimize_model(self):
        success_batch_size = min(len(self.success_memory), int(0.1 * BATCH_SIZE))
        normal_batch_size = BATCH_SIZE - success_batch_size
        if len(self.memory) < normal_batch_size:
            return
        if len(self.success_memory) == 0:
            return
        success_transitions = random.sample(self.success_memory, success_batch_size)
        transitions = random.sample(self.memory, normal_batch_size)
        success_batch = Transition(*zip(*success_transitions))
        batch = Transition(*zip(*transitions))
        # state_batch = torch.stack(batch.state).cuda().detach()
        # action_batch = torch.cat(batch.action).unsqueeze(1).detach()
        # reward_batch = torch.cat(batch.reward).detach()
        # next_state_batch = torch.stack(batch.next_state).cuda().detach()
        state_batch = torch.concatenate([torch.stack(batch.state).cuda().detach(), torch.stack(success_batch.state).cuda().detach()])
        action_batch = torch.concatenate([torch.cat(batch.action).unsqueeze(1).detach(), torch.cat(success_batch.action).unsqueeze(1).detach()])
        reward_batch = torch.concatenate([torch.cat(batch.reward).detach(), torch.cat(success_batch.reward).detach()])
        next_state_batch = torch.concatenate([torch.stack(batch.next_state).cuda().detach(), torch.stack(success_batch.next_state).cuda().detach()])
        
        state_action_values = self.policy_net(state_batch).cpu()
        state_action_values = state_action_values.gather(1, action_batch)
        next_state_values = self.target_net(next_state_batch).cpu()
        next_state_values = next_state_values.max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def draw(self, img, x, y, r):
        draw_img = img.clone()
        # print(draw_img.shape)
        for i in range(self.img_size):
            for j in range(self.img_size):
                if (i - x) ** 2 + (j - y) ** 2 <= r ** 2:
                    draw_img[0][0][i][j] = 0
        return draw_img
    
    def make_action(self, img, x, y, r, action):
        if action == 0:
            x = min(x + 1, self.img_size - 1)
        elif action == 1:
            x = max(x - 1, 0)
        elif action == 2:
            y = min(y + 1, self.img_size - 1)
        elif action == 3:
            y = max(y - 1, 0)
        elif action == 4:
            r = min(r + 1, self.img_size - 1)
        elif action == 5:
            r = max(r - 1, 1)
        elif action == 6:
            img = self.draw(img, x, y, r)
        return img, x, y, r
    
    def classifier_img(self, img):
        img = img.cuda()
        prob = self.classifier(img).cpu()
        img = img.cpu()
        return prob

    def train(self):
        k = 0
        for img, label in tqdm(self.train_loader):
            print(self.epsilon)
            if label != self.classifier_img(img).argmax():
                continue
            k += 1
            save_image(img, f"original/{k}.png")
            x, y, r = 0, 0, 0
            orig_img = img.clone()
            # label = label.cuda()
            onehot_label = torch.zeros(1, 10)
            onehot_label[0][label] = 1
            orig_prob = self.classifier_img(img)
            self.optimizer.zero_grad()
            state = torch.cat([img.view(-1), torch.tensor([x, y, r]), torch.tensor(self.history).float(), orig_prob.squeeze(), orig_prob.squeeze(), onehot_label.squeeze()])
            for j in range(self.max_iter):
                prob = self.classifier_img(img)
                action = self.select_action(state)
                self.action_lists.append(action)
                img, x, y, r = self.make_action(img, x, y, r, action)
                # save_img_mnist(img, f"current_img.png", x, y, r)
                reward = self.reward(img, orig_prob, orig_img, label)
                self.reward_lists.append(reward.item())
                self.history.append(action)
                next_state = torch.cat([img.view(-1).cpu(), torch.tensor([x, y, r]), torch.tensor(self.history).float(), prob.squeeze(), orig_prob.squeeze(), onehot_label.squeeze()])
                self.memory.append(Transition(state, torch.tensor([action]), next_state, torch.tensor([reward])))

                if self.num % 10 == 0:
                    loss = self.optimize_model()
                    self.loss_lists.append(loss)
                if self.num % 5 == 0:
                    self.epsilon = max(0.1, self.epsilon - 0.0001)
                if self.num % self.TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                if self.num % 1000 == 0:
                    torch.save(self.policy_net.state_dict(), r"model_0_trenvong_2.pth")
                    with open(r"actions.json", "w") as f:
                        json.dump(self.action_lists, f)
                    with open(r"rewards.json", "w") as f:
                        json.dump(self.reward_lists, f)
                    with open(r"losses.json", "w") as f:
                        json.dump(self.loss_lists, f)
                self.num += 1
                if label != prob.argmax():
                    self.success_memory.append(Transition(state, torch.tensor([action]), next_state, torch.tensor([reward])))
                    with open(r"logs.txt", "a") as f:
                        f.write(f"{k}: Step {j}: {label} -> {prob.argmax()} with {torch.norm(img - orig_img)}\n")
                    break
                state = next_state
            save_image(img, f"final/{k}.png")
        print("Finished training")
        torch.save(self.policy_net.state_dict(), r"model_0_trrenvong_2.pth")
        with open(r"actions.json", "w") as f:
            json.dump(self.action_lists, f)
        with open(r"rewards.json", "w") as f:
            json.dump(self.reward_lists, f)
        with open(r"losses.json", "w") as f:
            json.dump(self.loss_lists, f)
    def test(self):
        for img, label in tqdm(self.test_loader):
            x, y, r = 0, 0, 0
            orig_img = img.clone()
            img = img.cuda()
            prob = self.classifier(img)
            label = label.cuda()
            orig_img = img.clone()
            img.requires_grad = False
            self.optimizer.zero_grad()
            state = torch.cat([img.view(-1), torch.tensor([x, y, r]), torch.tensor(self.history).float().cuda()])
            for j in range(self.max_iter):
                prob = self.classifier(img)
                action = self.select_action(img.view(1, -1))
                self.action_lists.append(action)
                img, x, y, r = self.make_action(img, x, y, r, action)
                reward = self.reward(img, prob, orig_img)
                self.reward_lists.append(reward)
                self.history.append(action)
                next_state = torch.cat([img.view(-1), torch.tensor([x, y, r]), torch.tensor(self.history).float()])
                self.memory.append(Transition(state, torch.tensor([action]).cuda(), next_state, torch.tensor([reward])))
                state = next_state


                loss = self.optimize_model()
                self.loss_lists.append(loss)
                if j % self.TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
        print("Finished testing")
        with open(r"actions.json", "w") as f:
            json.dump(self.action_lists, f)
        with open(r"rewards.json", "w") as f:
            json.dump(self.reward_lists, f)
        with open(r"losses.json", "w") as f:
            json.dump(self.loss_lists, f)

if __name__ == "__main__":

    if not os.path.exists("original"):
        os.mkdir("original")
    if not os.path.exists("final"):
        os.mkdir("final")

    agent = Agent_Attack(device="cuda")
    agent.train()
    agent.test()


