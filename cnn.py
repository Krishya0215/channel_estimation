# lr=0.00001  cuda=5
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pydub import AudioSegment
from sklearn.metrics import mean_squared_error
import csv
import gc
import librosa

# Define the frequency cutoff
frequency_cutoff = 19000  # in Hz

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Dataset Class
class AudioDataset(Dataset):
    def __init__(self, signal_folder, double_folder, sample_rate=48000, duration=15):
        self.signal_folder = signal_folder
        self.double_folder = double_folder
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        self.cutoff_index = int(frequency_cutoff / (sample_rate / self.num_samples))  # Index for 19000 Hz

        self.signal_paths = [os.path.join(signal_folder, f) for f in os.listdir(signal_folder) if f.endswith('.wav')]
        self.double_paths = self.match_double_paths(self.signal_paths)

        if len(self.signal_paths) == 0 or len(self.double_paths) == 0:
            raise ValueError("No valid audio files found in signal or double folders.")

    def match_double_paths(self, signal_paths):
        double_paths = []
        for signal_path in signal_paths:
            signal_filename = os.path.basename(signal_path)
            double_filename = signal_filename[:-4] + '.wav'
            double_path = os.path.join(self.double_folder, double_filename)
            if os.path.exists(double_path):
                double_paths.append(double_path)
            else:
                raise ValueError(f"No matching file found in double folder for {signal_filename}")
        return double_paths

    def __len__(self):
        return len(self.signal_paths)

    def pad_or_truncate_waveform(self, waveform):
        if len(waveform) > self.num_samples:
            waveform = waveform[:self.num_samples]
        elif len(waveform) < self.num_samples:
            pad_length = self.num_samples - len(waveform)
            waveform = np.pad(waveform, (0, pad_length), mode='constant')
        return waveform

    def __getitem__(self, idx):
        signal_path = self.signal_paths[idx]
        double_path = self.double_paths[idx]

        signal_audio = AudioSegment.from_file(signal_path)
        signal_audio = signal_audio.set_frame_rate(self.sample_rate)
        signal_samples = np.array(signal_audio.get_array_of_samples()).astype(np.float32)

        if signal_audio.channels == 2:
            signal_samples = signal_samples.reshape(-1, 2)
            signal_waveform = signal_samples[:, 0]
        else:
            signal_waveform = signal_samples

        signal_waveform = self.pad_or_truncate_waveform(signal_waveform)
        signal_waveform = self.normalize_audio(signal_waveform)

        double_audio = AudioSegment.from_file(double_path)
        double_audio = double_audio.set_frame_rate(self.sample_rate)
        double_samples = np.array(double_audio.get_array_of_samples()).astype(np.float32)

        if double_audio.channels == 2:
            double_samples = double_samples.reshape(-1, 2)
            double_waveform = double_samples[:, 0]
        else:
            double_waveform = double_samples

        double_waveform = self.pad_or_truncate_waveform(double_waveform)
        double_waveform = self.normalize_audio(double_waveform)

        print('FFT前：signal_waveform.shape',signal_waveform.shape)
        print('FFT前：double_waveform.shape',double_waveform.shape)

        signal_spectrum = self.time_to_frequency(signal_waveform)
        double_spectrum = self.time_to_frequency(double_waveform)

        print(f"time_to_frequency后: signal_spectrum, {signal_spectrum.shape}")  # 打印输入维度
        print(f"time_to_frequency后: double_spectrum, {double_spectrum.shape}")  # 打印目标输出维度

        signal_spectrum = torch.from_numpy(signal_spectrum).float()
        double_spectrum = torch.from_numpy(double_spectrum).float()

        print(f"模型输入维度: {signal_spectrum.shape}")  # 打印输入维度
        print(f"目标输出维度: {double_spectrum.shape}")  # 打印目标输出维度

        return signal_spectrum, double_spectrum

    def normalize_audio(self, waveform):
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        return waveform

    def time_to_frequency(self, waveform):
        spectrum = np.fft.fft(waveform)
        print(f"FFT后的维度: {len(spectrum)}")
        # 裁掉 19000Hz 以下的频段
        spectrum = spectrum[self.cutoff_index:]  # Slice above the frequency cutoff
        print(f"裁剪后的频谱维度: {len(spectrum)}")
        real_part = np.real(spectrum)
        imag_part = np.imag(spectrum)
        combined = np.concatenate((real_part, imag_part))
        print(f"实部和虚部拼接后的维度: {len(combined)}")  # 打印拼接后的维度
        return combined.astype(np.float32)
#
# class CNN(nn.Module):
#     def __init__(self, input_size):
#         super(CNN, self).__init__()
#         self.input_size = input_size
#         self.num_conv_layers = 12
#         layers = []
#         in_channels = 1
#         out_channels = 2
#         for i in range(self.num_conv_layers):
#             layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
#             layers.append(nn.BatchNorm1d(out_channels))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(0.5))
#             layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
#             in_channels = out_channels
#             if (i + 1) % 4 == 0:
#                 out_channels *= 2
#         self.conv_layers = nn.Sequential(*layers)
#
#         with torch.no_grad():
#             print('self.input_size', self.input_size)
#             sample_input = torch.zeros(1, 1, self.input_size)
#             conv_output = self.conv_layers(sample_input)
#             print('conv_output', conv_output.shape)
#             self.feature_length = conv_output.shape[1] * conv_output.shape[2]
#             print('self.feature_length', self.feature_length)
#
#         # self.fc_layers = nn.Sequential(
#         #     nn.Linear(self.feature_length, 4096),
#         #     nn.ReLU(),
#         #     nn.Linear(4096, 2048),
#         #     nn.ReLU(),
#         #     nn.Linear(2048, 1024),
#         #     nn.ReLU(),
#         #     nn.Linear(1024, self.input_size)
#         # )
#         self.fc_layers = nn.Sequential(
#             nn.Linear(self.feature_length, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, self.input_size)
#         )
#
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc_layers(x)
#         return x


class CNN(nn.Module):
    def __init__(self, input_size):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.num_conv_layers = 12
        layers = []
        in_channels = 1
        out_channels = 2
        for i in range(self.num_conv_layers):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = out_channels
            if (i + 1) % 4 == 0:
                out_channels *= 2
        self.conv_layers = nn.Sequential(*layers)

        with torch.no_grad():
            print('self.input_size', self.input_size)
            sample_input = torch.zeros(1, 1, self.input_size)
            conv_output = self.conv_layers(sample_input)
            print('conv_output', conv_output.shape)
            self.feature_length = conv_output.shape[1] * conv_output.shape[2]
            print('self.feature_length', self.feature_length)

        self.output_layer = nn.Linear(self.feature_length, self.input_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv_layers(x)  # Apply convolution layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.output_layer(x)  # Output layer
        return x

# Testing Function
def test_model(input_audio_path, output_audio_path, model, device, sample_rate=48000, duration=15):
    signal_audio = AudioSegment.from_file(input_audio_path)
    signal_audio = signal_audio.set_frame_rate(sample_rate)

    if duration is not None:
        signal_audio = signal_audio[:duration * 1000]

    signal_samples = np.array(signal_audio.get_array_of_samples()).astype(np.float32)

    if signal_audio.channels == 2:
        signal_samples = signal_samples.reshape(-1, 2)
        signal_waveform = signal_samples[:, 0]
    else:
        signal_waveform = signal_samples

    max_val = np.max(np.abs(signal_waveform))
    if max_val > 0:
        signal_waveform = signal_waveform / max_val

    # 计算频率切割索引
    total_length = signal_waveform.shape[0]
    cutoff_index = int(frequency_cutoff / (sample_rate / total_length))

    spectrum = np.fft.fft(signal_waveform)
    # spectrum = spectrum[int(frequency_cutoff / (sample_rate / len(spectrum))):]

    # 仅保留 19000 Hz 以上的部分
    spectrum = spectrum[cutoff_index:]
    real_part = np.real(spectrum)
    imag_part = np.imag(spectrum)
    combined = np.concatenate((real_part, imag_part)).astype(np.float32)

    combined = torch.from_numpy(combined).float().to(device)

    model.eval()
    with torch.no_grad():
        output = model(combined.unsqueeze(0))
        output_value = output.cpu().numpy()[0]

    num_freq_bins = len(output_value) // 2
    output_real = output_value[:num_freq_bins]
    output_imag = output_value[num_freq_bins:]
    reconstructed_spectrum = output_real + 1j * output_imag

    reconstructed_waveform = np.fft.ifft(reconstructed_spectrum).real
    reconstructed_waveform = (reconstructed_waveform * 32767).astype(np.int16)
    output_audio = AudioSegment(
        reconstructed_waveform.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    output_audio.export(output_audio_path, format='wav')
    print(f"Output audio saved to {output_audio_path}")

# Main Training Code
signal_folder = 'signal'
double_folder = 'output5'
sample_rate = 48000
duration = 15
num_samples = sample_rate * duration
cutoff_index = int(frequency_cutoff / (sample_rate / num_samples))
input_size = (num_samples - cutoff_index) * 2
output_size = input_size
num_epochs = 1000000000000000000000000
best_loss = float('inf')
csv_filename = 'training_metrics_cnn.csv'


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = AudioDataset(signal_folder, double_folder, sample_rate=sample_rate, duration=duration)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    L = [(signal_batch, double_batch) for signal_batch, double_batch in data_loader]

    # L = [(signal_batch.to(device), double_batch.to(device)) for signal_batch, double_batch in data_loader]
    gc.collect()

    model = CNN(input_size=input_size).to(device)
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params

    total_params = count_parameters(model)
    print(f'总参数量: {total_params}')
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=1e-5)
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'Loss', 'MSE'])

        print("Training...")
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            # 用于计算MSE
            # all_targets = []
            # all_predictions = []

            for signal_batch, double_batch in L:
                signal_batch = signal_batch.to(device)
                double_batch = double_batch.to(device)

                optimizer.zero_grad()
                outputs = model(signal_batch)
                loss = criterion(outputs, double_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                # all_targets.append(double_batch.cpu().numpy())
                # all_predictions.append(outputs.cpu().detach().numpy())

                # Clear variables to free memory
                del signal_batch, double_batch, outputs
                gc.collect()

            avg_loss = total_loss / len(L)
            # all_targets = np.concatenate(all_targets)
            # all_predictions = np.concatenate(all_predictions)
            # mse = mean_squared_error(all_targets.flatten(), all_predictions.flatten())

            print(f'cnn, Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
            csv_writer.writerow([epoch + 1, avg_loss])

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), 'best_model_cnn.pth')

            gc.collect()

    # Testing the model
    input_audio_path = 'music/test_input.wav'
    output_audio_path = 'predict/test_output_cnn.wav'
    model.load_state_dict(torch.load('best_model_cnn.pth', map_location=device))
    test_model(input_audio_path, output_audio_path, model, device, sample_rate=sample_rate, duration=duration)
    gc.collect()


# # lr=0.00001  cuda=3
# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from pydub import AudioSegment
# from sklearn.metrics import mean_squared_error
# import csv
# import gc
# import random
# import librosa
#
# # Define the frequency cutoff
# frequency_cutoff = 19000  # in Hz
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#
# # Dataset Class
# class AudioDataset(Dataset):
#     def __init__(self, signal_folder, double_folder, sample_rate=48000, duration=15, subset_size=None):
#         self.signal_folder = signal_folder
#         self.double_folder = double_folder
#         self.sample_rate = sample_rate
#         self.duration = duration
#         self.num_samples = int(sample_rate * duration)
#         self.cutoff_index = int(frequency_cutoff / (sample_rate / self.num_samples))  # Index for 19000 Hz
#
#         self.signal_paths = [os.path.join(signal_folder, f) for f in os.listdir(signal_folder) if f.endswith('.wav')]
#         self.double_paths = self.match_double_paths(self.signal_paths)
#
#         if len(self.signal_paths) == 0 or len(self.double_paths) == 0:
#             raise ValueError("No valid audio files found in signal or double folders.")
#
#         # Support loading a subset of the dataset
#         if subset_size is not None:
#             indices = random.sample(range(len(self.signal_paths)), min(len(self.signal_paths), subset_size))
#             self.signal_paths = [self.signal_paths[i] for i in indices]
#             self.double_paths = [self.double_paths[i] for i in indices]
#
#     def match_double_paths(self, signal_paths):
#         double_paths = []
#         for signal_path in signal_paths:
#             signal_filename = os.path.basename(signal_path)
#             double_filename = signal_filename[:-4] + '.wav'
#             double_path = os.path.join(self.double_folder, double_filename)
#             if os.path.exists(double_path):
#                 double_paths.append(double_path)
#             else:
#                 raise ValueError(f"No matching file found in double folder for {signal_filename}")
#         return double_paths
#
#     def __len__(self):
#         return len(self.signal_paths)
#
#     def pad_or_truncate_waveform(self, waveform):
#         if len(waveform) > self.num_samples:
#             waveform = waveform[:self.num_samples]
#         elif len(waveform) < self.num_samples:
#             pad_length = self.num_samples - len(waveform)
#             waveform = np.pad(waveform, (0, pad_length), mode='constant')
#         return waveform
#
#     def __getitem__(self, idx):
#         signal_path = self.signal_paths[idx]
#         double_path = self.double_paths[idx]
#
#         signal_audio = AudioSegment.from_file(signal_path)
#         signal_audio = signal_audio.set_frame_rate(self.sample_rate)
#         signal_samples = np.array(signal_audio.get_array_of_samples()).astype(np.float32)
#
#         if signal_audio.channels == 2:
#             signal_samples = signal_samples.reshape(-1, 2)
#             signal_waveform = signal_samples[:, 0]
#         else:
#             signal_waveform = signal_samples
#
#         signal_waveform = self.pad_or_truncate_waveform(signal_waveform)
#         signal_waveform = self.normalize_audio(signal_waveform)
#
#         double_audio = AudioSegment.from_file(double_path)
#         double_audio = double_audio.set_frame_rate(self.sample_rate)
#         double_samples = np.array(double_audio.get_array_of_samples()).astype(np.float32)
#
#         if double_audio.channels == 2:
#             double_samples = double_samples.reshape(-1, 2)
#             double_waveform = double_samples[:, 0]
#         else:
#             double_waveform = double_samples
#
#         double_waveform = self.pad_or_truncate_waveform(double_waveform)
#         double_waveform = self.normalize_audio(double_waveform)
#
#         print('FFT前：signal_waveform.shape',signal_waveform.shape)
#         print('FFT前：double_waveform.shape',double_waveform.shape)
#
#         signal_spectrum = self.time_to_frequency(signal_waveform)
#         double_spectrum = self.time_to_frequency(double_waveform)
#
#         print(f"time_to_frequency后: signal_spectrum, {signal_spectrum.shape}")  # 打印输入维度
#         print(f"time_to_frequency后: double_spectrum, {double_spectrum.shape}")  # 打印目标输出维度
#
#         signal_spectrum = torch.from_numpy(signal_spectrum).float()
#         double_spectrum = torch.from_numpy(double_spectrum).float()
#
#         print(f"模型输入维度: {signal_spectrum.shape}")  # 打印输入维度
#         print(f"目标输出维度: {double_spectrum.shape}")  # 打印目标输出维度
#
#         return signal_spectrum, double_spectrum
#
#     def normalize_audio(self, waveform):
#         max_val = np.max(np.abs(waveform))
#         if max_val > 0:
#             waveform = waveform / max_val
#         return waveform
#
#     def time_to_frequency(self, waveform):
#         spectrum = np.fft.fft(waveform)
#         print(f"FFT后的维度: {len(spectrum)}")
#         # 裁掉 19000Hz 以下的频段
#         spectrum = spectrum[self.cutoff_index:]  # Slice above the frequency cutoff
#         print(f"裁剪后的频谱维度: {len(spectrum)}")
#         real_part = np.real(spectrum)
#         imag_part = np.imag(spectrum)
#         combined = np.concatenate((real_part, imag_part))
#         print(f"实部和虚部拼接后的维度: {len(combined)}")  # 打印拼接后的维度
#         return combined.astype(np.float32)
# #
# # class CNN(nn.Module):
# #     def __init__(self, input_size):
# #         super(CNN, self).__init__()
# #         self.input_size = input_size
# #         self.num_conv_layers = 12
# #         layers = []
# #         in_channels = 1
# #         out_channels = 2
# #         for i in range(self.num_conv_layers):
# #             layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
# #             layers.append(nn.BatchNorm1d(out_channels))
# #             layers.append(nn.ReLU())
# #             layers.append(nn.Dropout(0.5))
# #             layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
# #             in_channels = out_channels
# #             if (i + 1) % 4 == 0:
# #                 out_channels *= 2
# #         self.conv_layers = nn.Sequential(*layers)
# #
# #         with torch.no_grad():
# #             print('self.input_size', self.input_size)
# #             sample_input = torch.zeros(1, 1, self.input_size)
# #             conv_output = self.conv_layers(sample_input)
# #             print('conv_output', conv_output.shape)
# #             self.feature_length = conv_output.shape[1] * conv_output.shape[2]
# #             print('self.feature_length', self.feature_length)
# #
# #         # self.fc_layers = nn.Sequential(
# #         #     nn.Linear(self.feature_length, 4096),
# #         #     nn.ReLU(),
# #         #     nn.Linear(4096, 2048),
# #         #     nn.ReLU(),
# #         #     nn.Linear(2048, 1024),
# #         #     nn.ReLU(),
# #         #     nn.Linear(1024, self.input_size)
# #         # )
# #         self.fc_layers = nn.Sequential(
# #             nn.Linear(self.feature_length, 1024),
# #             nn.ReLU(),
# #             nn.Linear(1024, self.input_size)
# #         )
# #
# #     def forward(self, x):
# #         x = x.unsqueeze(1)
# #         x = self.conv_layers(x)
# #         x = x.view(x.size(0), -1)
# #         x = self.fc_layers(x)
# #         return x
#
#
# class CNN(nn.Module):
#     def __init__(self, input_size):
#         super(CNN, self).__init__()
#         self.input_size = input_size
#         self.num_conv_layers = 12
#         layers = []
#         in_channels = 1
#         out_channels = 2
#         for i in range(self.num_conv_layers):
#             layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
#             layers.append(nn.BatchNorm1d(out_channels))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(0.5))
#             layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
#             in_channels = out_channels
#             if (i + 1) % 4 == 0:
#                 out_channels *= 2
#         self.conv_layers = nn.Sequential(*layers)
#
#         with torch.no_grad():
#             print('self.input_size', self.input_size)
#             sample_input = torch.zeros(1, 1, self.input_size)
#             conv_output = self.conv_layers(sample_input)
#             print('conv_output', conv_output.shape)
#             self.feature_length = conv_output.shape[1] * conv_output.shape[2]
#             print('self.feature_length', self.feature_length)
#
#         self.output_layer = nn.Linear(self.feature_length, self.input_size)
#
#     def forward(self, x):
#         x = x.unsqueeze(1)  # Add channel dimension
#         x = self.conv_layers(x)  # Apply convolution layers
#         x = x.view(x.size(0), -1)  # Flatten
#         x = self.output_layer(x)  # Output layer
#         return x
#
# # Testing Function
# def test_model(input_audio_path, output_audio_path, model, device, sample_rate=48000, duration=15):
#     signal_audio = AudioSegment.from_file(input_audio_path)
#     signal_audio = signal_audio.set_frame_rate(sample_rate)
#
#     if duration is not None:
#         signal_audio = signal_audio[:duration * 1000]
#
#     signal_samples = np.array(signal_audio.get_array_of_samples()).astype(np.float32)
#
#     if signal_audio.channels == 2:
#         signal_samples = signal_samples.reshape(-1, 2)
#         signal_waveform = signal_samples[:, 0]
#     else:
#         signal_waveform = signal_samples
#
#     max_val = np.max(np.abs(signal_waveform))
#     if max_val > 0:
#         signal_waveform = signal_waveform / max_val
#
#     # 计算频率切割索引
#     total_length = signal_waveform.shape[0]
#     cutoff_index = int(frequency_cutoff / (sample_rate / total_length))
#
#     spectrum = np.fft.fft(signal_waveform)
#     # spectrum = spectrum[int(frequency_cutoff / (sample_rate / len(spectrum))):]
#
#     # 仅保留 19000 Hz 以上的部分
#     spectrum = spectrum[cutoff_index:]
#     real_part = np.real(spectrum)
#     imag_part = np.imag(spectrum)
#     combined = np.concatenate((real_part, imag_part)).astype(np.float32)
#
#     combined = torch.from_numpy(combined).float().to(device)
#
#     model.eval()
#     with torch.no_grad():
#         output = model(combined.unsqueeze(0))
#         output_value = output.cpu().numpy()[0]
#
#     num_freq_bins = len(output_value) // 2
#     output_real = output_value[:num_freq_bins]
#     output_imag = output_value[num_freq_bins:]
#     reconstructed_spectrum = output_real + 1j * output_imag
#
#     reconstructed_waveform = np.fft.ifft(reconstructed_spectrum).real
#     reconstructed_waveform = (reconstructed_waveform * 32767).astype(np.int16)
#     output_audio = AudioSegment(
#         reconstructed_waveform.tobytes(),
#         frame_rate=sample_rate,
#         sample_width=2,
#         channels=1
#     )
#     output_audio.export(output_audio_path, format='wav')
#     print(f"Output audio saved to {output_audio_path}")
#
# # Main Training Code
# signal_folder = 'signal'
# double_folder = 'output5'
# sample_rate = 48000
# duration = 15
# num_samples = sample_rate * duration
# cutoff_index = int(frequency_cutoff / (sample_rate / num_samples))
# input_size = (num_samples - cutoff_index) * 2
# output_size = input_size
# num_epochs = 1000000000000000000000000
# best_loss = float('inf')
# csv_filename = 'training_metrics_cnn.csv'
#
#
# if __name__ == '__main__':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
#
#     # dataset = AudioDataset(signal_folder, double_folder, sample_rate=sample_rate, duration=duration)
#     # data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
#
#     # # Pre-load data to GPU memory
#     # L = [(signal_batch.to(device), double_batch.to(device)) for signal_batch, double_batch in data_loader]
#     # gc.collect()
#
#     model = CNN(input_size=input_size).to(device)
#     # Calculate and print total parameters
#     # def count_parameters(model):
#     #     total_params = 0
#     #     for param in model.parameters():
#     #         total_params += param.numel()
#     #     return total_params
#     def count_parameters(model):
#         total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         return total_params
#
#     total_params = count_parameters(model)
#     print(f'总参数量: {total_params}')
#
#     criterion = nn.MSELoss()
#     optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=1e-5)
#
#     subset_size = 16  # Number of samples to load per batch
#     reload_interval = 20  # Number of epochs before reloading data
#
#     with open(csv_filename, mode='w', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)
#         csv_writer.writerow(['Epoch', 'Loss', 'MSE'])
#
#         print("Training...")
#         for epoch in range(num_epochs):
#             # Reload dataset periodically
#             if epoch % reload_interval == 0 or epoch == 0:
#                 dataset = AudioDataset(signal_folder, double_folder, sample_rate=sample_rate,
#                                        duration=duration, subset_size=subset_size)
#                 data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
#                 print(f"Dataset reloaded at epoch {epoch + 1}")
#
#             model.train()
#             total_loss = 0
#             # 用于计算MSE
#             # all_targets = []
#             # all_predictions = []
#
#             for signal_batch, double_batch in data_loader:
#                 signal_batch = signal_batch.to(device)
#                 double_batch = double_batch.to(device)
#
#                 optimizer.zero_grad()
#                 outputs = model(signal_batch)
#                 loss = criterion(outputs, double_batch)
#                 loss.backward()
#                 optimizer.step()
#
#                 total_loss += loss.item()
#                 # all_targets.append(double_batch.cpu().numpy())
#                 # all_predictions.append(outputs.cpu().detach().numpy())
#
#                 # Clear variables to free memory
#                 del signal_batch, double_batch, outputs
#                 gc.collect()
#
#             avg_loss = total_loss / len(L)
#             # all_targets = np.concatenate(all_targets)
#             # all_predictions = np.concatenate(all_predictions)
#             # mse = mean_squared_error(all_targets.flatten(), all_predictions.flatten())
#
#             print(f'cnn, Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
#             csv_writer.writerow([epoch + 1, avg_loss])
#
#             if avg_loss < best_loss:
#                 best_loss = avg_loss
#                 torch.save(model.state_dict(), 'best_model_cnn.pth')
#
#             gc.collect()
#
#     # Testing the model
#     input_audio_path = 'music/test_input.wav'
#     output_audio_path = 'predict/test_output_cnn.wav'
#     model.load_state_dict(torch.load('best_model_cnn.pth', map_location=device))
#     test_model(input_audio_path, output_audio_path, model, device, sample_rate=sample_rate, duration=duration)
#     gc.collect()
