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

        self.grid_size = 2
        self.grid_map = self.img_size // self.grid_size

        self.input_size = self.img_size * self.img_size * 2
        self.output_size = self.grid_map * self.grid_map
        
        self.policy_net =  DQN_Conv(self.input_size, self.output_size).cuda().train()
        
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

        self.target_net = DQN_Conv(self.input_size, self.output_size).cuda()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=1e-6)
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
            return torch.tensor(2).float()
        # return prob_diff - 2*img_diff
        return torch.tensor(0).float()
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            action1 = random.randint(0, self.output_size - 1)
            action2 = action1
            while action2 == action1:
                action2 = random.randint(0, self.output_size - 1)
            return action1, action2
        with torch.no_grad():
            prob = self.policy_net(state.cuda()).cpu()
            # if random.random() < 0.01:
            #     print(prob)
            action1 = prob.argmax().item()
            action2 = prob.argmin().item()
            return action1, action2
        
    def optimize_model(self):
        success_batch_size = min(len(self.success_memory), int(0.1 * BATCH_SIZE))
        normal_batch_size = BATCH_SIZE - success_batch_size
        # normal_batch_size = BATCH_SIZE
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
        # done_batch = torch.tensor(batch.done).detach()
        state_batch = torch.concatenate([torch.stack(batch.state).cuda().detach(), torch.stack(success_batch.state).cuda().detach()])
        action_batch = torch.concatenate([torch.cat(batch.action).unsqueeze(1).detach(), torch.cat(success_batch.action).unsqueeze(1).detach()])
        reward_batch = torch.concatenate([torch.cat(batch.reward).detach(), torch.cat(success_batch.reward).detach()])
        next_state_batch = torch.concatenate([torch.stack(batch.next_state).cuda().detach(), torch.stack(success_batch.next_state).cuda().detach()])
        done_batch = torch.concatenate([torch.tensor(batch.done).detach(), torch.tensor(success_batch.done).detach()])
        
        state_action_values = self.policy_net(state_batch.squeeze(1)).cpu()
        state_action_values = state_action_values.gather(1, action_batch)
        next_state_values = self.target_net(next_state_batch.squeeze(1)).cpu()
        next_state_values = next_state_values.max(1)[0].detach()
        expected_state_action_values = torch.where(done_batch, reward_batch, (next_state_values * self.GAMMA) + reward_batch)

        # print(expected_state_action_values)

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    
    def make_action(self, orgi_img, img, grid1, grid2, e):
        # random_grid = torch.rand((2, 2)) * 2 - 1
        random_grid = torch.ones((2, 2)) * -1
        mask = img - orgi_img

        mask_temp = mask.clone()

        grid_map2 = mask_temp[0, 0, grid2//self.grid_size:grid2//self.grid_size + 2, grid2%self.grid_size:grid2%self.grid_size + 2]

        avail_distance = e**2 - torch.norm(mask_temp).item()**2 + torch.norm(grid_map2).item()**2
        
        avail_distance = max(0, avail_distance)
        avail_distance = min(avail_distance, e**2)

        add_grid = random_grid / torch.norm(random_grid) * avail_distance

        mask_temp[0,0,grid1//self.grid_map*self.grid_size:grid1//self.grid_map*self.grid_size + 2, grid1%self.grid_map*self.grid_size:grid1%self.grid_map*self.grid_size + 2] += add_grid
        mask_temp[0,0,grid2//self.grid_map*self.grid_size:grid2//self.grid_map*self.grid_size + 2, grid2%self.grid_map*self.grid_size:grid2%self.grid_map*self.grid_size + 2] = 0

        new_img = orgi_img + mask_temp
        new_img = torch.clamp(new_img, 0, 1)

        return new_img
    
    def classifier_img(self, img):
        img = img.cuda()
        prob = self.classifier(img).cpu()
        img = img.cpu()
        return prob

    def train(self):
        k = 0
        total_success = 0
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
            state = torch.cat([img, orig_img], dim=1)
            for j in range(self.max_iter):
                prob = self.classifier_img(img)
                action1, action2 = self.select_action(state)
                self.action_lists.append((action1, action2))
                img = self.make_action(orig_img, img, action1, action2, 2)
                # save_img_mnist(img, f"current_img.png", x, y, r)
                reward = self.reward(img, orig_prob, orig_img, label)
                self.reward_lists.append(reward.item())
                # self.history.append(action)
                next_state = torch.cat([img.cpu(), orig_img.cpu()], dim=1)
                # if j == self.max_iter - 1 and label == prob.argmax():
                #     self.success_memory.append(Transition(state, torch.tensor([action1]), next_state, torch.tensor([-2]), True))
                # else:
                #     self.memory.append(Transition(state, torch.tensor([action1]), next_state, torch.tensor([reward]), label != prob.argmax()))

                if self.num % 30 == 0:
                    loss = self.optimize_model()
                    self.loss_lists.append(loss)
                if self.num % 150 == 0:
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
                    total_success += 1
                    print(f"Step {j}: {label} -> {prob.argmax()} with {torch.norm(img - orig_img)}, {total_success}/{k}, {orig_prob[0][label].item()} -> {prob[0][label].item()}, {prob.max().item()}")
                    self.success_memory.append(Transition(state, torch.tensor([action1]), next_state, torch.tensor([reward]), True))
                    with open(r"logs_succ.txt", "a") as f:
                        f.write(f"{k}: Step {j}: {label} -> {prob.argmax()} with {torch.norm(img - orig_img)}\n")
                    break

                if j == self.max_iter - 2:
                    self.success_memory.append(Transition(state, torch.tensor([action1]), next_state, torch.tensor([-2]), True))
                    break

                self.memory.append(Transition(state, torch.tensor([action1]), next_state, torch.tensor([reward]), False))

                state = next_state
            save_image(img, f"final/{k}.png")
            with open(r"logs.txt", "a") as f:
                f.write(f"{k}: {label} -> {prob.argmax()} with {torch.norm(img - orig_img)}, prob_label: {orig_prob[0][label].item()} -> {prob[0][label].item()}, prob_max: {prob.max().item()}\n")
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


