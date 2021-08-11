import torch
from torch import nn
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch.optim as optim
import os
torch.manual_seed(0)

dir_name = './generated_images'
os.makedirs(dir_name, exist_ok=True)

def show_tensor_images(image_tensor, filename, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    save_image(image_grid, os.path.join(dir_name, filename))
    # plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    # plt.show()

def crop(image, new_shape):
    '''
    Function for cropping an image tensor: Given an image tensor and the new shape,
    crops to the center pixels.
    Parameters:
        image: image tensor of shape (batch size, channels, height, width)
        new_shape: a torch.Size object with the shape you want x to have
    '''
    middle_height = image.shape[2] // 2
    middle_width = image.shape[3] // 2
    starting_height = middle_height - round(new_shape[2] / 2)
    final_height = starting_height + new_shape[2]
    starting_width = middle_width - round(new_shape[3] / 2)
    final_width = starting_width + new_shape[3]
    cropped_image = image[:, :, starting_height:final_height, starting_width:final_width]
    return cropped_image

class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs two convolutions followed by a max pool operation.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels * 2)
        self.use_bn = use_bn
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock: 
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x

class ExpandingBlock(nn.Module):
    '''
    ExpandingBlock Class:
    Performs an upsampling, a convolution, a concatenation of its two inputs,
    followed by two more convolutions with optional dropout
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=2)
        self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=2, padding=1)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x, skip_con_x):
        '''
        Function for completing a forward pass of ExpandingBlock: 
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
            skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        x = self.upsample(x)
        x = self.conv1(x)
        skip_con_x = crop(skip_con_x, x.shape)
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x

class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a U-Net - 
    maps each pixel to a pixel with the correct number of output dimensions
    using a 1x1 convolution.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        '''
        Function for completing a forward pass of FeatureMapBlock: 
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x

class UNet(nn.Module):
    '''
    UNet Class
    A series of 4 contracting blocks followed by 4 expanding blocks to 
    transform an input image into the corresponding paired image, with an upfeature
    layer at the start and a downfeature layer at the end.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels, hidden_channels=32):
        super(UNet, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_dropout=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, use_dropout=True)
        self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout=True)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.contract5 = ContractingBlock(hidden_channels * 16)
        self.contract6 = ContractingBlock(hidden_channels * 32)
        self.expand0 = ExpandingBlock(hidden_channels * 64)
        self.expand1 = ExpandingBlock(hidden_channels * 32)
        self.expand2 = ExpandingBlock(hidden_channels * 16)
        self.expand3 = ExpandingBlock(hidden_channels * 8)
        self.expand4 = ExpandingBlock(hidden_channels * 4)
        self.expand5 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        '''
        Function for completing a forward pass of UNet: 
        Given an image tensor, passes it through U-Net and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.contract5(x4)
        x6 = self.contract6(x5)
        x7 = self.expand0(x6, x5)
        x8 = self.expand1(x7, x4)
        x9 = self.expand2(x8, x3)
        x10 = self.expand3(x9, x2)
        x11 = self.expand4(x10, x1)
        x12 = self.expand5(x11, x0)
        xn = self.downfeature(x12)
        return self.sigmoid(xn)

# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED CLASS: Discriminator
# Discriminator generates an output matrix of values between 0 and 1 
# indicating the fakeness or realness of the images
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake. 
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels, hidden_channels=8):
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        #### START CODE HERE ####
        self.final = nn.Conv2d(hidden_channels * 16, 1, kernel_size=1)
        #### END CODE HERE ####

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn

# Example code 
# input_channels = 3
# hidden_channels = 32
# output_channels = 3
# unet = UNet(input_channels, output_channels, hidden_channels)
# input_image = torch.randn(4, input_channels, 512, 512)
# print(f'Input image dimension : {input_image.shape}')
# output_image = unet(input_image)
# print(f'Output image dimension : {output_image.shape}')

# disc = Discriminator(2*input_channels)
# predictions = disc(output_image, input_image)
# print(f'Predictions dimension : {predictions.shape}')

# There will be two loss criterions
# Adversarial loss which is the BCELoss 
# PixelDistance loss which is the L1 loss 
adversarial_criterion = nn.BCEWithLogitsLoss()
l1_criterion = nn.L1Loss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# the input and the expected output are concatenated together width wise
batch_size=4
lr = 2e-4
epochs = 100

input_channels = 3
output_channels = 3
target_shape = 256
lambda_recon = 200
pretrained = True
saved_checkpoint = 'gen_pretrained_9_12.46.pth'

# better weight initialization for the generator and discriminator 

# initialize the generator and discriminiator model 
gen = UNet(input_channels, output_channels)
gen = nn.DataParallel(gen)
gen.to(device)
gen_opt = optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator(input_channels+output_channels)
disc = nn.DataParallel(disc)
disc.to(device)
disc_opt = optim.Adam(disc.parameters(), lr=lr)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

# load the pretrained weights if available 
if pretrained: 
    # saved_checkpoint = 'gen_loaded.pth'
    loaded_state = torch.load(saved_checkpoint)
    # load the generator and discriminator's model and optimizer weights
    gen.load_state_dict(loaded_state['gen'])
    gen_opt.load_state_dict(loaded_state['gen_opt'])
    disc.load_state_dict(loaded_state['disc'])
    disc_opt.load_state_dict(loaded_state['disc_opt'])
else:
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)


# load the dataset 
transform = transforms.Compose([transforms.ToTensor(),])
dataset = datasets.ImageFolder("maps", transform=transform)
dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

def train():

    # start the training, we have to convert satellite imagery into map routes 
    avg_gen_loss = 0
    avg_disc_loss = 0
    disp_step = 200
    current_step = 0
    min_gen_loss = 1000

    for epoch in range(epochs):
        for index, (image, _) in enumerate(dataloader):
            condition = image[:, :, :, :image.shape[3] // 2]
            # change the dimension of the condition image
            condition = nn.functional.interpolate(condition, size=target_shape)
            real = image[:, :, :, image.shape[3] // 2 : ]
            # change the dimension of the expected output image
            real = nn.functional.interpolate(real, size=target_shape)

            condition = condition.to(device)
            real = real.to(device)

            # generate generator's predictions conditioned on the condition 
            fake = gen(condition)
            # print(f'Fake image dimension : {fake.shape}')
            # zero the gradients 
            disc_opt.zero_grad()
            # compute the discriminator's predictions on the generated images
            disc_fake_pred = disc(fake.detach(), condition)
            disc_adv_loss_fake = adversarial_criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_pred = disc(real, condition)
            disc_adv_loss_real = adversarial_criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_adv_loss_fake + disc_adv_loss_real) / 2
            disc_loss.backward(retain_graph=True) # retain graph because we want to compute the loss for generator
            disc_opt.step()

            # update the generator 
            gen_opt.zero_grad()
            fake = gen(condition)
            disc_fake_pred = disc(fake, condition)
            # compute the adversarial loss, 
            # the gen should fool the disc into believing that the images produced by it are real
            gen_disc_adv_loss = adversarial_criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            # compute the reconstruction or pixel distance loss 
            # between the generated output and the expected (real) output
            recon_loss = l1_criterion(fake, real)
            gen_loss = gen_disc_adv_loss + lambda_recon * recon_loss 
            gen_loss.backward()
            gen_opt.step()

            current_step += 1

            avg_disc_loss += disc_loss.item() / disp_step
            avg_gen_loss += gen_loss.item() / disp_step

            if current_step % disp_step == 0:
                print(f'Epoch : {epoch}, step : {current_step}, avg gen loss : {avg_gen_loss}, avg disc loss : {avg_disc_loss}')
                
                # calculate the generator loss 
                if avg_gen_loss < min_gen_loss:
                    print('Saving generator checkpoints')
                    checkpoint_path = 'gen_pretrained_{}_{}.pth'.format(epoch, round(avg_gen_loss, 2))
                    # save the model parameters 
                    torch.save({
                        'epoch': epoch,
                        'gen': gen.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc': disc.state_dict(),
                        'disc_opt': disc_opt.state_dict(),
                        'gen_loss': avg_gen_loss,
                        'disc_loss': avg_disc_loss
                        }, checkpoint_path)

                avg_gen_loss = 0
                avg_disc_loss = 0

# run the inference and generate a few images
def eval():
    gen.eval()
    disc.eval()
    image, _ = next(iter(dataloader))
    sat_image = image[:, :, :, :image.shape[3] // 2]
    sat_image = nn.functional.interpolate(sat_image, size=target_shape)
    output = image[:, :, :, image.shape[3] // 2:]
    output = nn.functional.interpolate(output, size=target_shape)
    # generate the generated images for the input
    fake = gen(sat_image)
    show_tensor_images(sat_image, 'satellite_images.png', size=(input_channels, target_shape, target_shape))
    show_tensor_images(output, 'map_images.png', size=(output_channels, target_shape, target_shape))
    show_tensor_images(fake, 'generated_map_images.png', size=(output_channels, target_shape, target_shape))

eval()