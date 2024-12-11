import torch
from collections import defaultdict
from tqdm import tqdm


def extract_save_features(loader: torch.utils.data.DataLoader,
                        model: torch.nn.Module,
                        device: torch.device,
                        filename: str):
    """The function extract_save_features saves a dictionary. Within the context of this section, the dictionnary that is saved will have :

For the training set :
the key ‘features’ with torch tensor of shape (5717, 512, 7, 7)
the key ‘bboxes’ with torch tensor of shape (5717,4)
the key ‘labels’ with torch tensor of shape (5717)
For the validation set :
the key ‘features’ with torch tensor of shape (5823, 512, 7, 7)
the key ‘bboxes’ with torch tensor of shape (5823,4)
the key ‘labels’ with torch tensor of shape (5823)"""
    all_features = []
    all_targets = defaultdict(list)

    with torch.no_grad():
        model.eval()
        for (inputs, targets) in tqdm(loader):

            inputs = inputs.to(device=device)

            # Compute the forward propagation through the body
            # just to extract the features
            all_features.append(model(inputs))

            for k, v in targets.items():
                all_targets[k].append(v)

        for k, v in all_targets.items():
            all_targets[k] = torch.squeeze(torch.cat(v, 0))
        all_features = torch.squeeze(torch.cat(all_features , 0))
    print(all_features.shape)
    print("The features that are saved are {} features maps of size {} x {}, with {} channels".format(all_features.shape[0], all_features.shape[2], all_features.shape[3], all_features.shape[1]))
    for k, v in all_targets.items():
        print("The entry {} have shape {}".format(k, v.shape))

    torch.save(dict([("features", all_features)] + list(all_targets.items())), filename)


def train(model: torch.nn.Module,
          loader:torch.utils.data.DataLoader,
          optimizer:torch.optim.Optimizer,
          device: torch.device):
    """
        Train a model for one epoch, iterating over the loader
        using the f_loss to compute the loss and the optimizer
        to update the parameters of the model.

        Arguments :
        model      -- A torch.nn.Module object
        loader     -- A torch.utils.data.DataLoader
        optimizer  -- A torch.optim.Optimzer object
        device     -- The device to use for the computation CPU or GPU

        Returns :

    """


    # We enter train mode. This is useless for the linear model
    # but is important for layers such as dropout, batchnorm, ...

    bbox_loss  = torch.nn.SmoothL1Loss()
    bbox_reg_loss = torch.nn.L1Loss(reduction='sum')
    class_loss = torch.nn.CrossEntropyLoss()

    alpha_bbox = 20.0

    model.train()
    N = 0
    regression_loss, correct = 0.0, 0
    for i, (inputs, bboxes, labels) in enumerate(loader):

        inputs, bboxes, labels = inputs.to(device=device), bboxes.to(device=device), labels.to(device=device)

        # Compute the forward propagation
        outputs = model(inputs)

        # On the first steps, b_loss around 0.1, c_loss around 20.0
        b_loss = alpha_bbox * bbox_loss(outputs[0], bboxes)
        c_loss = class_loss(outputs[1], labels)


        # Accumulate the number of processed samples
        N += inputs.shape[0] # Updates the count of processed samples.

        # For the total loss
        regression_loss += bbox_reg_loss(outputs[0], bboxes).item()/4.0

        # For the total accuracy
        predicted_targets = outputs[1].argmax(dim=1)
        correct += (predicted_targets == labels).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        b_loss.backward()
        c_loss.backward()
        try:
            model.penalty().backward()
        except AttributeError:
            pass
        optimizer.step()

        # Display status
        progress_bar(i, len(loader), msg = "bbox loss : {:.4f}, classification Acc : {:.4f}".format(regression_loss/N, correct/N))
    return regression_loss/N, correct/N



def test(model, loader, device):
    """
    Test a model by iterating over the loader

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        device    -- device to be used for the computation (CPU or GPU)

    Returns :

        A tuple with the mean loss and mean accuracy

    """
    bbox_reg_loss = torch.nn.L1Loss(reduction='sum')
    class_loss = torch.nn.CrossEntropyLoss(reduction='sum')

    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..
        model.eval()
        N = 0
        regression_loss, correct = 0.0, 0
        for i, (inputs, bboxes, labels) in enumerate(loader):

            inputs, bboxes, labels = inputs.to(device=device), bboxes.to(device=device), labels.to(device=device)

            outputs = model(inputs)

            b_loss = bbox_reg_loss(outputs[0], bboxes)/4.0

            N += inputs.shape[0]

            # For the total bbox loss
            regression_loss += b_loss.item()

            # For the accuracy
            predicted_targets = outputs[1].argmax(dim=1)
            correct += (predicted_targets == labels).sum().item()
        return regression_loss/N, correct/N
    """Key Differences from train Function:

No Gradient Calculation: torch.no_grad() is used to disable gradient computation during evaluation, saving computation and memory.
No Parameter Updates: The optimizer is not used in the test function, as the goal is only to evaluate the model's performance, not to train it further.
Focus on Metrics: The test function focuses on calculating and returning the average loss and accuracy, which are used to assess the model's performance."""