import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from my_data import meta_dataloaders, visualize_random_episodes
from model import ProtoNet, resnet, SimpleEmbedding
from train import meta_one_epoch, train, report_train_result
from callbacks import EarlyStopping
from my_util import DEVICE

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    loaders = meta_dataloaders("pairs.jsonl",
                               num_episodes=10000,)
    # visualize_random_episodes(loaders[0].dataset)

    MODEL_NAME = "meta_ghibli_resnet6"

    embedding_net = resnet(6, 64).to(DEVICE)
    info = summary(embedding_net, verbose=0)
    with open(f"{MODEL_NAME}_summary.txt", "w") as f:
        f.write(str(info))

    model = ProtoNet(num_ways=2, num_shots=3,
                     embedding_net=embedding_net).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    es = EarlyStopping(patience=5, restore_best_weights=True)
    result = train(model, loaders, criterion, optimizer,
                   meta_one_epoch, DEVICE,
                   num_epochs=10000, callbacks=[es])

    torch.save(model.state_dict(), f"{MODEL_NAME}.pth")
    print("Model saved!")
    report_train_result(result, MODEL_NAME)
