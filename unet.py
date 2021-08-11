import torch 
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Contracting(nn.Module):
	def __init__(self, input_channels):
		super(Contracting, self).__init__()
		self.conv1 = nn.Conv2d(input_channels, input_channels*2, kernel_size=3)
		self.conv2 = nn.Conv2d(input_channels*2, input_channels*2, kernel_size=3)
		self.activation = nn.ReLU()
		self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
		# define the models inside the init function

	def forward(self, x):
		x = self.conv1(x)
		x = self.activation(x)
		x = self.conv2(x)
		x = self.activation(x)
		x = self.pooling(x)

		return x

# Expanding path 
class Expanding(nn.Module):
	def __init__(self, input_channels):
		super(Expanding, self).__init__()
		self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=2)
		self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3)
		self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=3)
		self.activation = nn.ReLU()

	def forward(self, x, residual_connection):
		x = self.upsampling(x)
		print(f'shape after upsample : {x.shape}')
		x = self.conv1(x)
		skip_connection = crop(residual_connection, x.shape) # return residual of size x 
		x = torch.cat([x, skip_connection], dim=1) # concatenate along channel dimension 
		x = self.conv2(x)
		x = self.activation(x)
		x = self.conv3(x)
		x = self.activation(x)
		return x

# crop the input image to the dimension specified in shape
def crop(residual, output_shape):
	middle_height = residual.shape[2] // 2
	start_height = middle_height - output_shape[2] // 2
	end_height = start_height + output_shape[2]
	middle_width = residual.shape[3] // 2
	start_width = middle_width - output_shape[3] // 2
	end_width = start_width + output_shape[3]
	return residual[:, :, start_height:end_height, start_width:end_width]

class FeatureMap(nn.Module):
	def __init__(self, input_channels, output_channels):
		super(FeatureMap, self).__init__()
		self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

	def forward(self, x):
		return self.conv(x)

def test_contracting_block():
	input_channels = 32
	input_image = torch.randn(10, input_channels, 572, 572).to(device)
	contracting_block = Contracting(input_channels).to(device)
	output_image = contracting_block(input_image)
	print(output_image.to('cpu').shape)

# test_contracting_block()
# U-Net used for image translation, primarily used for generating segmentation maps 
class Unet(nn.Module):
	def __init__(self, input_channels, hidden_channels, output_channels):
		super(Unet, self).__init__()
		# input -> 3 x 572 x 572
		self.upfeature = FeatureMap(input_channels, hidden_channels) # 32 x 572 x 572
		self.contracting1 = Contracting(hidden_channels) 
		self.contracting2 = Contracting(hidden_channels*2)
		self.contracting3 = Contracting(hidden_channels*4)
		self.contracting4 = Contracting(hidden_channels*8)
		self.contracting5 = Contracting(hidden_channels*16)
		self.expanding1 = Expanding(hidden_channels*32)
		self.expanding2 = Expanding(hidden_channels*16)
		self.expanding3 = Expanding(hidden_channels*8)
		self.expanding4 = Expanding(hidden_channels*4)
		self.downfeature = FeatureMap(hidden_channels*2, output_channels)

	def forward(self, x):
		upfeature = self.upfeature(x)
		contract1 = self.contracting1(upfeature)
		contract2 = self.contracting2(contract1)
		contract3 = self.contracting3(contract2)
		contract4 = self.contracting4(contract3)
		contract5 = self.contracting5(contract4)
		expand1 = self.expanding1(contract5, contract4)
		expand2 = self.expanding2(expand1, contract3)
		expand3 = self.expanding3(expand2, contract2)
		expand4 = self.expanding4(expand3, contract1)
		downfeature = self.downfeature(expand4)

		return downfeature

# define the model 
input_channels = 3
hidden_channels = 32
output_channels = 2
unet = Unet(input_channels, hidden_channels, output_channels)
input_image = torch.randn(4, 3, 572, 572)
output_image = unet(input_image)
print(f'Input shape : {input_image.shape}')
print(f'Output shape : {output_image.shape}')