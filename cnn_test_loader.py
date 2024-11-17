def test_model_using_loader(test_loader, model, device, output_folder='test_outputs', sample_rate=48000, duration=15):
    model.eval()  # Set model to evaluation mode
    criterion = nn.MSELoss()  # Loss function

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_loss = 0

    with torch.no_grad():
        for idx, (signal_batch, double_batch) in enumerate(test_list):
            signal_batch = signal_batch.to(device)
            double_batch = double_batch.to(device)

            # Forward pass
            outputs = model(signal_batch)

            # Calculate loss
            loss = criterion(outputs, double_batch)
            total_loss += loss.item()

            # Process the output and save as audio
            for i in range(signal_batch.size(0)):  # Iterate over each sample in the batch
                # Get the current signal and model output
                signal = signal_batch[i].cpu().numpy()
                output = outputs[i].cpu().numpy()

                # Reconstruct the signal and output as audio

                # Compute the cutoff index for the frequency domain
                cutoff_index = int(frequency_cutoff / (sample_rate / signal.shape[0]))

                # Create the frequency spectrum from the signal
                signal_spectrum = np.fft.fft(signal)
                signal_spectrum = signal_spectrum[cutoff_index:]  # Keep only higher frequencies

                # Reconstruct the spectrum from the model output
                num_freq_bins = len(output) // 2
                output_real = output[:num_freq_bins]
                output_imag = output[num_freq_bins:]
                reconstructed_spectrum = output_real + 1j * output_imag

                # Concatenate the low frequencies and model output
                reconstructed_spectrum = np.concatenate([np.zeros(cutoff_index), reconstructed_spectrum])

                # Convert back to time-domain signal using IFFT
                reconstructed_waveform = np.fft.ifft(reconstructed_spectrum).real
                reconstructed_waveform = (reconstructed_waveform * 32767).astype(np.int16)

                # Create an AudioSegment from the waveform
                output_audio = AudioSegment(
                    reconstructed_waveform.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=2,
                    channels=1
                )

                # Save the output audio to a file
                output_audio_path = os.path.join(output_folder, f'output_{idx * test_loader.batch_size + i}.wav')
                output_audio.export(output_audio_path, format='wav')

                print(f"Output audio saved to {output_audio_path}")

    avg_test_loss = total_loss / len(test_loader)
    print(f'Average Test Loss: {avg_test_loss:.4f}')


# Main testing code using test_loader
if __name__ == '__main__':
    # Load the best model
    model.load_state_dict(torch.load('best_model_cnn.pth', map_location=device))

    # Test using the test_loader
    test_model_using_loader(test_loader, model, device, output_folder='test_outputs', sample_rate=sample_rate, duration=duration)
