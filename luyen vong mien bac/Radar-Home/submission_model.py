  
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label
def find_largest_connected_component(predictions):

    # Set the top 5 rows to zero
    predictions[:, :5, :] = 0

    # Set the bottom 5 rows to zero
    predictions[:, -5:, :] = 0

    # Set the first 5 columns to zero
    predictions[:, :, :15] = 0

    # Set the last 5 columns to zero
    predictions[:, :, -15:] = 0
    # Convert predictions to numpy array
    predictions_np = predictions.cpu().numpy()
    return predictions_np
    # Initialize an array to store the largest component
    # largest_component = np.zeros_like(predictions_np)

    # for i in range(predictions_np.shape[0]):  # Iterate over batch
    #     # Label connected components
    #     labeled_array, num_features = label(predictions_np[i])

    #     # Find the largest component
    #     if num_features > 0:
    #         largest_component_size = 0
    #         largest_component_label = 0
    #         for label_num in range(1, num_features + 1):
    #             component_size = np.sum(labeled_array == label_num)
    #             if component_size > largest_component_size:
    #                 largest_component_size = component_size
    #                 largest_component_label = label_num

    #         # Set the largest component in the output
    #         largest_component[i] = (labeled_array == largest_component_label)

    # return torch.tensor(largest_component, dtype=torch.float32).to(predictions.device)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # Encoder
        self.enc_conv1 = self.conv_block(6, 16)
        self.enc_conv2 = self.conv_block(16, 32)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = self.conv_block(32, 64)

        # Decoder
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv1 = self.conv_block(96, 32)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv2 = self.conv_block(48, 16)

        # Output layer
        self.out_conv = nn.Conv2d(16, 2, kernel_size=1)

    def conv_block(self, in_channels, out_channels, dropout_rate=0.5):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),  # Use LeakyReLU
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)  # Use LeakyReLU
        )
    def dilate_label1_regions(self, predictions):
        # 获取当前预测的label 1 mask [B, H, W]
        pred_mask = torch.argmax(predictions, dim=1)

        # 创建非对称膨胀核（5x3大小）
        kernel = torch.zeros((1, 1, 3, 5), device=predictions.device)  # [1,1,H,W]
        kernel[0, 0, 1, :] = 1  # 中心行全1（横向延伸2像素）

        # 对每个样本进行处理
        dilated_masks = []
        for i in range(pred_mask.shape[0]):
            mask = pred_mask[i].float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

            # 应用膨胀（padding=2横向，padding=1纵向）
            dilated = F.conv2d(mask, kernel, padding=(1, 2))  # (padH, padW)
            dilated = (dilated > 0).float()
            dilated_masks.append(dilated.squeeze())

        dilated_mask = torch.stack(dilated_masks)  # [B,H,W]

        # 更新预测结果
        new_label1_mask = (dilated_mask == 1) & (pred_mask == 0)
        processed_output = predictions.clone()
        processed_output[:, 1][new_label1_mask] = 1.0   # 强制新区域预测为1
        processed_output[:, 0][new_label1_mask] = -1.0  # 抑制背景通道

        return processed_output
    def forward(self, x, mode = 'test'):
        padding = (5, 6, 3, 3)  
        x = F.pad(x, padding, mode='constant', value=0).cuda()

        # Encoder
        x1 = self.enc_conv1(x)
        x2 = self.pool(x1)
        x2 = self.enc_conv2(x2)
        x3 = self.pool(x2)

        # Bottleneck
        x3 = self.bottleneck(x3)

        # Decoder
        x4 = self.upsample1(x3)
        x4 = torch.cat([x4, x2], dim=1)  # Skip connection
        x4 = self.dec_conv1(x4)

        x5 = self.upsample2(x4)
        x5 = torch.cat([x5, x1], dim=1)  # Skip connection
        x5 = self.dec_conv2(x5)
        # Output layer
        x_out = self.out_conv(x5)
        # Crop the output to the desired size (181, 50)
        x_out = x_out[:, :, :50, :181]
        if mode == 'test':
            predictions = torch.argmax(x_out, dim=1)
            largest_component = find_largest_connected_component(predictions)
            mask = largest_component == 0
            x_out[:, 1, :, :][mask] = float('-inf')
            #visualize_prediction(largest_component)
            #visualize_prediction(predictions)
            #visualize_prediction(torch.argmax(x_out, dim=1))
            kernel_weights = torch.zeros((2, 2, 5, 5), device=x_out.device)  # [out_ch, in_ch, H, W]

        # Middle row is 1 for both input and output channels
            kernel_weights[:, :, 2, :] = 1  # Set middle row to 1 for all channels
            kernel_weights[:, :, 1:4, 2] = 1
        # Apply the convolution with padding=2 to maintain spatial dimensions
            x_out = self.process_label1_regions(x_out)
            x_out = self.dilate_label1_regions(x_out)
            return x_out
        if mode == 'test2':
            predictions = torch.argmax(x_out, dim=1)
            largest_component = find_largest_connected_component(predictions)
            mask = largest_component == 0
            x_out[:, 1, :, :][mask] = float('-inf')
            #visualize_prediction(largest_component)
            #visualize_prediction(predictions)
            #visualize_prediction(torch.argmax(x_out, dim=1))
            kernel_weights = torch.zeros((2, 2, 5, 5), device=x_out.device)  # [out_ch, in_ch, H, W]

        # Middle row is 1 for both input and output channels
            kernel_weights[:, :, 2, :] = 1  # Set middle row to 1 for all channels
            kernel_weights[:, :, 1:4, 2] = 1
        # Apply the convolution with padding=2 to maintain spatial dimensions
            x_out = self.process_label1_regions(x_out)
            return x_out
        return x_out
    def process_label1_regions(self, predictions, threshold_ratio=0.5):
        # Get binary mask for label 1
        pred_mask = torch.argmax(predictions, dim=1)  # [B, H, W]
        label1_mask = (pred_mask == 1).cpu().numpy()  # Convert to numpy for scipy

        processed_output = predictions.clone()

        for i in range(predictions.shape[0]):  # Process each sample in batch
            # Label connected components
            labeled_array, num_features = label(label1_mask[i])

            if num_features == 0:
                continue  # No label 1 regions

            # Calculate bounding box widths for each component
            bbox_widths = []
            for label_num in range(1, num_features + 1):
                rows, cols = np.where(labeled_array == label_num)
                if len(rows) == 0:
                    bbox_widths.append(0)
                    continue
                min_row, max_row = np.min(rows), np.max(rows)
                min_col, max_col = np.min(cols), np.max(cols)
                width = max_col - min_col + 1  # +1 because both ends are inclusive
                bbox_widths.append(width)

            max_width = np.max(bbox_widths)
            threshold = max_width * threshold_ratio

            # Create mask for narrow regions to remove
            remove_mask = np.zeros_like(label1_mask[i], dtype=bool)

            for label_num in range(1, num_features + 1):
                if bbox_widths[label_num - 1] < threshold:
                    remove_mask |= (labeled_array == label_num)

            # Set narrow regions to label 0 by setting label 1 channel to -inf
            if remove_mask.any():
                processed_output[i, 1][torch.from_numpy(remove_mask).to(predictions.device)] = float('-inf')

        return processed_output
