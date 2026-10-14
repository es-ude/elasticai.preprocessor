#ifndef SPIKE_DETECTION_H
#define SPIKE_DETECTION_H

void Processing::detect_spikes(double filtered_value, uint32_t sampleIdx, int channel) {
    // save last spikes occurrence in given channel
    static std::vector<uint32_t> last_spike_events(cfg.n_channel);

    // After 5 seconds, if the value deviates much from the current standard deviation, a spike is detected
    if(filtered_value < -5 * runningStdDev_calcs[channel]->getStandardDeviation() and sampleIdx > 5*cfg.sampling_rate) {

        // if the spike is at least 10 samples after the last spike in this channel
        if(sampleIdx > last_spike_events[channel]+10) {
            spike_events.push_back(SpikeEvent(channel,sampleIdx));
            last_spike_events[channel] = sampleIdx;

        }
    }
}

std::vector<double> Processing::extract_waveform(SpikeEvent *spike_event,int frame_start, int frame_end, int pos_in_win) {

    std::vector<double> spike_waveform;
    static int window_size = cfg.buffer.window_size;
    static int spike_cut_out_len = cfg.model.input_size;

    // trivial case when the frame is completely within the window
    if(frame_start >= 0 and frame_end <= window_size - spike_cut_out_len/2) {
        for (int i=0; i < cfg.model.input_size; i++) {
            spike_waveform.emplace_back(window[pos_in_win+i-spike_cut_out_len/2].sample[spike_event->channel]);
        }
        return spike_waveform;
    }

    // frame is at the start of the current frame
    if(frame_start < 0) {
        auto prev_window = window_buffer.back();
        for (int i=abs(frame_start); i > 0; i--) {
            spike_waveform.emplace_back(prev_window.data[window_size-i].sample[spike_event->channel]);
        }
        for (int i=0; i < spike_cut_out_len+frame_start; i++) {
            spike_waveform.emplace_back(window[i].sample[spike_event->channel]);
        }
        return spike_waveform;
    }

    // if the spikeEvent is from the end previous window
    if(spike_event->isOld == true) {
        auto prev_window = window_buffer.back();
        // frame start is in the old window
        for (int i=frame_start; i < window_size; i++) {
            spike_waveform.emplace_back(prev_window.data[i].sample[spike_event->channel]);
        }
        // frame end is in the current window
        for(int i=0; i <abs(window_size-frame_end); i++) {
            spike_waveform.emplace_back(window[i].sample[spike_event->channel]);
        }
        return spike_waveform;
    }

    // if the spike event is at the end of the current window + not from the previous window
    if(frame_end>=window_size and spike_event->isOld == false) {
        spike_event->isOld = true,
        spike_events.push_back(*spike_event);
    }

    return spike_waveform;
}

class OnlineStdDev {
private:
    double mean;       // Running mean
    double m2;         // Sum of squared differences from the mean
    int count;         // Number of samples processed

public:
    OnlineStdDev();

    // Update the statistics with a new sample
    void update(double sample);

    // Get the current standard deviation
    [[nodiscard]] double getStandardDeviation() const;
};


#endif
