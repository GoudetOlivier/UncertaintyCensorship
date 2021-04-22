import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import torch

# from sksurv.nonparametric import kaplan_meier_estimator
# from sksurv.metrics import concordance_index_censored


from torchvision import datasets, transforms




transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)




trainloader = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=1000, shuffle=True)

# print(trainset.size())
# print(valset.size())

# for images, labels in trainloader:
#     # Redimensionnement de l'image 28*28 en vecteur de taille 784
#     images = images.view(images.shape[0], -1)

print(trainloader.get(0).size())


for images, labels in trainloader:
    x_train = images

    print(x_train.size())

# x_test = x_test.astype(np.float32) / 255.
#
# y_train = y_train.astype(np.int32)
# y_test = y_test.astype(np.int32)


def make_risk_score_for_groups(y: np.ndarray,
                               n_groups: int = 4,
                               seed: int = 89) -> Tuple[pd.DataFrame, np.ndarray]:
    rnd = np.random.RandomState(seed)

    # assign class labels `y` to one of `n_groups` risk groups
    classes = np.unique(y)
    group_assignment = {}
    group_members = {}
    groups = rnd.randint(n_groups, size=classes.shape)
    for label, group in zip(classes, groups):
        group_assignment[label] = group
        group_members.setdefault(group, []).append(label)

    # assign risk score to each class label in `y`
    risk_per_class = {}
    for label in classes:
        group_idx = group_assignment[label]
        group = group_members[group_idx]
        label_idx = group.index(label)
        group_size = len(group)

        # allow risk scores in each group to vary slightly
        risk_score = np.sqrt(group_idx + 1e-4) * 1.75
        risk_score -= (label_idx - (group_size // 2)) / 25.
        risk_per_class[label] = risk_score

    assignment = pd.concat((
        pd.Series(risk_per_class, name="risk_score"),
        pd.Series(group_assignment, name="risk_group")
    ), axis=1).rename_axis("class_label")

    risk_scores = np.array([risk_per_class[yy] for yy in y])
    return assignment, risk_scores


risk_score_assignment, risk_scores = make_risk_score_for_groups(y)

risk_score_assignment.round(3)
