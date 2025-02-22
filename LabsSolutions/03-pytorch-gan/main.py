#!/usr/bin/env python3
# coding: utf-8

"""
Implements GAN training inspired from :

    - Radford et al(2015) Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
    - Saliman et al (2016) Improved techniques for training GANs
      https://github.com/openai/improved-gan
"""

# Standard imports
import argparse
import logging
import sys
import os
from typing import Callable, Dict

# External imports
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import deepcs.display
from deepcs.fileutils import generate_unique_logpath
import tqdm
import onnxruntime as ort

# Local imports
import data
import models


def label_smooth(labels, amplitude, num_target_classes):
    """
    Apply label smoothing to labels
    Consider labels is (B, )

    Args:
        labels: a (B, ) tensor with the the hard labels
        amplitude [0, 1]: the maximum amount of the probability mass to assign to the non class labels
        num_target_classes: the number of hard labels
    Returns:
        a tensor (B, num_target_classes) with the smoothed labels
    """
    # Fill everywhere  rand(0, 1) * amplitude / (num_target_classes - 1)
    smoothed_labels = (
        amplitude
        * torch.rand((labels.shape[0], num_target_classes), device=labels.device)
        / (num_target_classes - 1.0)
    )

    # For the correct class, we set its value to (1 - amplitude*rand(0, 1))
    # The probabilities do not sum to 1 but, almost
    smoothed_labels[:, labels] = 1 - amplitude * torch.rand(
        (labels.shape[0],), device=labels.device
    )  # (B, num_target_classes)

    # Normalize the soft labels so that they sum to 1, as a discrete distribution over the num_target_classes
    # labels
    norm_factor = smoothed_labels.sum(axis=1)  # (B, )
    smoothed_labels /= norm_factor[:, None]
    return smoothed_labels


def train(args):
    """
    Training of the algorithm
    """
    logger = logging.getLogger(__name__)
    logger.info("Training")

    # Parameters
    dataset = args.dataset
    dataset_root = args.dataset_root
    nthreads = args.nthreads
    batch_size = args.batch_size
    dropout = args.dropout
    debug = args.debug
    base_lr = args.base_lr
    wdecay = args.wdecay
    lblsmooth = args.lblsmooth
    dnoise = args.dnoise
    num_epochs = args.num_epochs
    discriminator_base_c = args.discriminator_base_c
    generator_base_c = args.generator_base_c
    latent_size = args.latent_size
    num_classes = (
        2  # TODO: we could be using the true labels for training the discriminator
    )
    sample_nrows = 8
    sample_ncols = 8

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # Dataloaders
    train_loader, valid_loader, img_shape = data.get_dataloaders(
        dataset_root=dataset_root,
        cuda=use_cuda,
        batch_size=batch_size,
        n_threads=nthreads,
        dataset=dataset,
        small_experiment=debug,
    )

    # Model definition
    model = models.GAN(
        img_shape,
        dropout,
        discriminator_base_c,
        dnoise,
        num_classes,
        latent_size,
        generator_base_c,
    )
    model.to(device)

    # Optimizers
    critic = model.discriminator
    generator = model.generator
    ######################
    # START CODING HERE ##
    ######################
    # Step 1 - Define the optimizer for the critic
    # @TEMPL@optim_critic = None
    # @SOL
    if wdecay == 0:
        logger.info("No weight decay")
        optim_critic = optim.Adam(critic.parameters(), lr=base_lr, betas=(0.5, 0.999))
    else:
        optim_critic = optim.AdamW(
            critic.parameters(), lr=base_lr, weight_decay=wdecay, betas=(0.5, 0.999)
        )

    # SOL@
    # Step 2 - Define the optimizer for the generator
    # @TEMPL@optim_generator = None
    # @SOL
    optim_generator = optim.Adam(generator.parameters(), lr=base_lr, betas=(0.5, 0.999))
    # SOL@

    # Step 3 - Define the loss (it must embed the sigmoid)
    # @TEMPL@loss = None
    loss = torch.nn.BCEWithLogitsLoss()  # @SOL@

    ####################
    # END CODING HERE ##
    ####################

    # Callbacks
    summary_text = (
        "## Summary of the model architecture\n"
        + f"{deepcs.display.torch_summarize(model)}\n"
    )
    summary_text += "\n\n## Executed command :\n" + "{}".format(" ".join(sys.argv))
    summary_text += "\n\n## Args : \n {}".format(args)

    logger.info(summary_text)

    if args.logdir is None:
        logdir = generate_unique_logpath("./logs", "gan")
    else:
        logdir = args.logdir
        if not os.path.exists(logdir):
            os.makedirs(logdir)
    tensorboard_writer = SummaryWriter(log_dir=logdir, flush_secs=5)
    tensorboard_writer.add_text(
        "Experiment summary", deepcs.display.htmlize(summary_text)
    )

    with open(os.path.join(logdir, "summary.txt"), "w") as f:
        f.write(summary_text)

    save_path = os.path.join(logdir, "generator.pt")

    logger.info(f">>>>> Results saved in {logdir}")

    # Note: the validation data are only real images, hence the metrics below
    val_fmetrics = {
        "accuracy": lambda real_probas: (real_probas > 0.5).double().mean(),
        "loss": lambda real_probas: -real_probas.log().mean(),
    }

    # Define a fixed noise used for sampling
    fixed_noise = torch.randn(sample_nrows * sample_ncols, latent_size).to(device)

    # Generate few samples from the initial generator
    model.eval()
    fake_images = model.generator(X=fixed_noise)
    fake_images = (fake_images * data._IMG_STD + data._IMG_MEAN).clamp(0, 1.0)
    grid = torchvision.utils.make_grid(fake_images, nrow=sample_nrows, normalize=True)
    tensorboard_writer.add_image("Generated", grid, 0)

    imgpath = logdir + "/images/"
    if not os.path.exists(imgpath):
        os.makedirs(imgpath)
    torchvision.utils.save_image(grid, imgpath + "images-0000.png")

    # Training loop
    for e in range(num_epochs):

        tot_dploss = tot_dnloss = tot_gloss = 0
        critic_paccuracy = critic_naccuracy = 0
        generator_accuracy = 0
        Ns = 0

        model.train()
        for ei, (X, _) in enumerate(tqdm.tqdm(train_loader)):

            # X is a batch of real data
            X = X.to(device)
            bi = X.shape[0]

            pos_labels = torch.ones((bi,)).long().to(device)
            neg_labels = torch.zeros((bi,)).long().to(device)

            ######################
            # START CODING HERE ##
            ######################

            # -- Discriminator training --

            # Step 1 - Forward pass for training the discriminator
            # Ganhacks #4: Use separate batchs for real and fake data
            # @TEMPL@real_logits, _ = None
            # @TEMPL@fake_logits, _ = None
            # @SOL
            real_logits, _ = model(
                X,
                None,
            )
            # SOL@
            fake_logits, _ = model(None, bi)  # @SOL@

            # Step 1 - Compute the real and fake labels for the discriminator
            # @TEMPL@discriminator_real_labels = None # (bi,)
            # @TEMPL@discriminator_fake_labels = None # (bi,)

            # @SOL
            discriminator_real_labels = pos_labels
            discriminator_fake_labels = neg_labels
            # SOL@

            # Ganhacks #6: use soft labels
            # Apply a random label smoothing for this minibatch
            discriminator_smoothed_real_labels = label_smooth(
                discriminator_real_labels, lblsmooth, num_classes
            )  # (B, num_target_classes)
            discriminator_smoothed_fake_labels = label_smooth(
                discriminator_fake_labels, lblsmooth, num_classes
            )  # (B, num_target_classes)

            # Step 2 - Compute the loss of the critic
            # @TEMPL@D_ploss = None
            # @TEMPL@D_nloss = None
            # @TEMPL@Dloss = None + None
            # @SOL
            D_ploss = loss(real_logits, discriminator_smoothed_real_labels)
            D_nloss = loss(fake_logits, discriminator_smoothed_fake_labels)
            Dloss = 0.5 * (D_ploss + D_nloss)
            # SOL@

            # Step 3 - Reinitialize the gradient accumulator of the critic
            # @TEMPL@None
            optim_critic.zero_grad()  # @SOL@

            # Step 4 - Perform the backward pass on the loss
            # @TEMPL@None
            Dloss.backward()  # @SOL@

            # Step 5 - Update the parameters of the critic
            # @TEMPL@None
            optim_critic.step()  # @SOL@

            ####################
            # END CODING HERE ##
            ####################

            # Computing of metrics for the discriminator
            # real_probs = torch.softmax(real_logits, dim=1)[:, 1]
            # fake_probs = torch.softmax(fake_logits, dim=1)[:, 0]
            critic_paccuracy += (real_logits[:, 1] > real_logits[:, 0]).sum().item()
            critic_naccuracy += (fake_logits[:, 0] > fake_logits[:, 1]).sum().item()
            tot_dploss += bi * D_ploss.item()
            tot_dnloss += bi * D_nloss.item()

            ######################
            # START CODING HERE ##
            ######################

            # -- Generator training --
            # The principle is to generator samples
            # and update the parameters of the generator only so that the frozen discriminator
            # considers the generated samples as real

            # Step 1 - Forward pass for training the generator
            # @TEMPL@fake_logits, _ = None
            fake_logits, _ = model(None, bi)  # @SOL@

            # Step 1 - Compute the fake labels for the generator
            # The generator wants his generated images to be positive
            # @TEMPL@generator_fake_labels = None # (bi,)
            generator_fake_labels = pos_labels  # @SOL@

            # Ganhacks #6: use soft labels
            # Apply a random label smoothing for this minibatch
            generator_fake_labels = label_smooth(
                generator_fake_labels, 0.0, num_classes
            )

            # Step 2 - Compute the loss of the generator
            # @TEMPL@Gloss = None
            Gloss = loss(fake_logits, generator_fake_labels)  # @SOL@

            # Step 3 - Reinitialize the gradient accumulator of the generator
            # @TEMPL@None
            optim_generator.zero_grad()  # @SOL@

            # Step 4 - Perform the backward pass on the loss
            # @TEMPL@None
            Gloss.backward()  # @SOL@

            # Step 5 - Update the parameters of the generator
            # @TEMPL@None
            optim_generator.step()  # @SOL@
            ####################
            # END CODING HERE ##
            ####################

            # Update the metrics for the generator
            tot_gloss += bi * Gloss.item()
            generator_accuracy += (fake_logits[:, 1] > fake_logits[:, 0]).sum().item()

            # Accumulate the number of samples we saw
            Ns += bi

        # Normalize the metrics by the total number of samples
        critic_paccuracy /= Ns
        critic_naccuracy /= Ns
        tot_dploss /= Ns
        tot_dnloss /= Ns
        generator_accuracy /= Ns
        tot_gloss /= Ns

        # Evaluate the metrics on the validation set
        val_metrics = evaluate(model, device, valid_loader, val_fmetrics)

        logger.info(
            f"[Epoch {e+1}] \n"
            f"  - Critic BCELoss averaged on the real samples : {tot_dploss:.4f} , \n"
            f"  - Critic p-accuracy : Fraction of real samples considered as real by the discriminator : {critic_paccuracy:.2f}, \n"
            f"  - Critic BCELoss averaged on the fake samples : {tot_dnloss:.4f} , \n"
            f"  - Critic n-accuracy : Fraction of fake samples considered as fake by the discriminator :{critic_naccuracy:.2f},\n"
            f"  - Critic v-loss : CE Loss on real samples from the test fold :  {val_metrics['loss']:.4f}; \n"
            f"  - Critic accuracy on real samples from the validation fold : {val_metrics['accuracy']:.2f}, \n"
            f"  - Generator BCELoss averaged on the fake samples : {tot_gloss:.4f} \n"
            f"  - Generator accurarcy : fraction of generated samples considered as real by the discriminator {generator_accuracy:.2f}\n"
        )

        tensorboard_writer.add_scalar(
            "Critic p-loss : Critic BCELoss averaged on the real samples",
            tot_dploss,
            e + 1,
        )
        tensorboard_writer.add_scalar(
            "Critic p-accuracy : Fraction of real samples considered as real by the discriminator",
            critic_paccuracy,
            e + 1,
        )
        tensorboard_writer.add_scalar(
            "Critic n-loss : Critic BCELoss averaged on the fake samples",
            tot_dnloss,
            e + 1,
        )
        tensorboard_writer.add_scalar(
            "Critic n-accuracy : Fraction of fake samples considered as fake by the discriminator",
            critic_naccuracy,
            e + 1,
        )
        tensorboard_writer.add_scalar(
            "Critic v-loss : BCE Loss on real samples from the validation fold",
            val_metrics["loss"],
            e + 1,
        )
        tensorboard_writer.add_scalar(
            "Critic v-accuracy : Fraction of real samples from the validation fold considered as real by the discriminator",
            val_metrics["accuracy"],
            e + 1,
        )
        tensorboard_writer.add_scalar(
            "Generator BCELoss averaged on the fake samples", tot_gloss, e + 1
        )
        tensorboard_writer.add_scalar(
            "Generator accurarcy : fraction of generated samples considered as real by the discriminator",
            generator_accuracy,
            e + 1,
        )

        # Generate few samples from the generator
        model.eval()
        fake_images = model.generator(X=fixed_noise)
        # Unscale the images
        fake_images = (fake_images * data._IMG_STD + data._IMG_MEAN).clamp(0, 1.0)
        grid = torchvision.utils.make_grid(fake_images, nrow=sample_nrows)
        tensorboard_writer.add_image("Generated", grid, e + 1)
        torchvision.utils.save_image(grid, imgpath + f"images-{e+1:04d}.png")

        X, _ = next(iter(train_loader))
        real_images = X[: (sample_nrows * sample_ncols), ...]
        real_images = (real_images * data._IMG_STD + data._IMG_MEAN).clamp(0, 1.0)
        grid = torchvision.utils.make_grid(real_images, nrow=sample_nrows)
        tensorboard_writer.add_image("Real", grid, e + 1)
        # torchvision.utils.save_image(grid, imgpath + f"real-{e+1:04d}.png")

        # We save the generator
        logger.info(f"Generator saved at {save_path}")
        torch.save(model.generator, save_path)

        # Important: ensure the model is in eval mode before exporting !
        # the graph in train/test mode is not the same
        model.eval()
        dummy_input = torch.zeros((1, latent_size), device=device)
        torch.onnx.export(
            model.generator,
            dummy_input,
            logdir + "/generator.onnx",
            verbose=False,
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )  # At least opset 11 is required otherwise it seems nn.UpSample is not correctly handled
        logger.info(f"Generator onnx saved at {logdir}/generator.onnx")


def evaluate(
    model: torch.nn.Module,
    device: torch.device,
    loader: torch.utils.data.DataLoader,
    metrics: Dict[str, Callable],
):
    """
    Compute the averaged metrics given in the dictionnary.
    The dictionnary metrics gives a function to compute the metrics on a
    minibatch and averaged on it.
    """
    model.eval()

    tot_metrics = {m_name: 0.0 for m_name in metrics}
    Ns = 0
    for (inputs, targets) in loader:

        # Move the data to the GPU if required
        inputs, targets = inputs.to(device), targets.to(device)

        batch_size = inputs.shape[0]

        # Forward pass
        logits, _ = model(inputs, None)
        probas = logits.sigmoid()

        # Compute the metrics
        for m_name, m_f in metrics.items():
            tot_metrics[m_name] += batch_size * m_f(probas).item()
        Ns += batch_size

    # Size average the metrics
    for m_name, m_v in tot_metrics.items():
        tot_metrics[m_name] = m_v / Ns

    return tot_metrics


class ONNXWrapper:
    def __init__(self, use_cuda, modelpath):
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
        providers = []
        if use_cuda:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.inference_session = ort.InferenceSession(modelpath, providers=providers)
        input_shapes = self.inference_session.get_inputs()[
            0
        ].shape  # ['batch', latent_size]
        self.latent_size = input_shapes[1]

    def eval(self):
        # ONNX model cannot be switched from train to test
        pass

    def train(self):
        # ONNX model cannot be switch from test to train
        pass

    def __call__(self, torchX):
        output = self.inference_session.run(
            None, {self.inference_session.get_inputs()[0].name: torchX.cpu().numpy()}
        )[0]

        return torch.from_numpy(output).to(self.device)


def generate(args):
    """
    Function to generate new samples from the generator
    using a pretrained network
    """

    # Parameters
    modelpath = args.modelpath
    assert modelpath is not None

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    ######################
    # START CODING HERE ##
    ######################
    # Step 1 - Reload the generator using the ONNXWrapper class
    # @TEMPL@generator = None
    generator = ONNXWrapper(use_cuda, modelpath)  # @SOL@

    # Put the model in evaluation mode (due to BN and Dropout)
    generator.eval()

    # Generate some samples
    sample_nrows = 1
    sample_ncols = 8

    # Step 2 - Generate a noise vector, normaly distributed
    #          of shape (sample_nrows * sample_ncol, generator.latent_size)
    # @TEMPL@z = None
    # @SOL
    z = torch.randn(sample_nrows * sample_ncols, generator.latent_size).to(device)
    # SOL@

    # Step 3 - Forward pass through the generator
    #          The output is (B, 1, 28, 28)
    # The ONNXWrapper.__call__ implementation allows to forward propagate through ONNXWrapper
    # as you would do with normal nn.Module
    # @TEMPL@fake_images = None
    fake_images = generator(z)  # @SOL@

    # Denormalize the result
    fake_images = fake_images * data._IMG_STD + data._IMG_MEAN
    ####################
    # END CODING HERE ##
    ####################

    grid = torchvision.utils.make_grid(fake_images, nrow=sample_ncols, normalize=True)
    torchvision.utils.save_image(grid, "generated1.png")
    logger.info("Image generated1.png generated")

    # @SOL
    # Interpolate in the laten space
    N = 20
    z = torch.zeros((N, N, generator.latent_size)).to(device)
    # Generate the 3 corner samples
    z[0, 0, :] = torch.randn(generator.latent_size)
    z[-1, 0, :] = torch.randn(generator.latent_size)
    z[0, -1, :] = torch.randn(generator.latent_size)
    di = z[-1, 0, :] - z[0, 0, :]
    dj = z[0, -1, :] - z[0, 0, :]
    for i in range(0, N):
        for j in range(0, N):
            z[i, j, :] = z[0, 0, :] + i / (N - 1) * di + j / (N - 1) * dj
    fake_images = generator(z.reshape(N**2, -1))
    fake_images = fake_images * data._IMG_STD + data._IMG_MEAN
    grid = torchvision.utils.make_grid(fake_images, nrow=N, normalize=True)
    torchvision.utils.save_image(grid, "generated2.png")
    logger.info("Interpolation image generated2.png generated")
    # SOL@


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "generate"])

    # Data parameters
    parser.add_argument(
        "--dataset",
        choices=["MNIST", "FashionMNIST", "EMNIST", "SVHN", "CelebA"],
        help="Which dataset to use",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        help="The root dir where the datasets are stored",
        default=data._DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        help="The number of threads to use " "for loading the data",
        default=6,
    )
    parser.add_argument(
        "--logdir",
        type=str,
        help="The logdir in which to save the assets of the experiments",
        default=None,
    )

    # Training parameters
    parser.add_argument(
        "--num_epochs", type=int, help="The number of epochs to train for", default=200
    )
    parser.add_argument(
        "--batch_size", type=int, help="The size of a minibatch", default=256
    )
    parser.add_argument(
        "--base_lr", type=float, help="The initial learning rate to use", default=0.0002
    )
    parser.add_argument(
        "--wdecay", type=float, help="The weight decay used for the critic", default=0.0
    )
    parser.add_argument(
        "--lblsmooth", type=float, help="The amplitude of label smoothing", default=0.2
    )
    parser.add_argument(
        "--dnoise",
        type=float,
        help="Variance of input discriminator random noise",
        default=0.1,
    )

    parser.add_argument(
        "--debug", action="store_true", help="Whether to use small datasets"
    )

    # Architectures
    parser.add_argument(
        "--discriminator_base_c",
        type=int,
        help="The base number of channels for the discriminator",
        default=32,
    )
    parser.add_argument(
        "--generator_base_c",
        type=int,
        help="The base number of channels for the generator",
        default=256,
    )
    parser.add_argument(
        "--latent_size", type=int, help="The dimension of the latent space", default=100
    )

    # Regularization
    parser.add_argument(
        "--dropout",
        type=float,
        help="The probability of zeroing in the discriminator",
        default=0.5,
    )

    # For the generation
    parser.add_argument(
        "--modelpath",
        type=str,
        help="The path to the pt file of the generator to load",
        default=None,
    )

    args = parser.parse_args()

    eval(f"{args.command}(args)")
