import numpy as np
import h5py

""" 
basic flow of the program:
- we start by parsing events from the 4d array to a structured numpy array
    - these arrays are in the form (height, width, time, polarity)
    - ideally want to output in the form (x, y, t, pol)
- we then filter out hot pixels by counting the number of events per pixel and removing pixels with counts above a threshold
    - a "hot pixel" is defined as a pixel that constantly generates event due to some issue with the camera, like defects, glare, etc.. creating noise
- mask_region_filter is used to remove events in a specified rectangular region
    - (this is taken from DHP19, thought it might be useful because of the shape issue we had earlier)
- normalize_image_3sigma is used to normalize the event counts using 3-sigma normalization
    - this is done to normalize the data for lack of better words
    - the normalization is done by subtracting the mean and dividing by 3 times the standard deviation
    - we do this to ensure that the patterns in the data are more visible, and to make the data more interpretable
- process_events_to_frames is used to process the events into normalized frames with a fixed event count
    - this is done by sorting the events chronologically, then creating a frame with a fixed number of events
    - we have it at 7500 events/frame right now, we can change this if need be, i picked 7500 because DHP19 used 7500
"""


def parse_events(event_array):
    """Parse events from (W, H, T, 2) array to structured numpy array"""
    W, H, T, _ = event_array.shape
    events = np.zeros(W * H * T, dtype=[
        ('x', 'i4'), ('y', 'i4'), ('t', 'u8'), ('pol', 'i1')
    ])
    count = 0

    for x in range(W):
        for y in range(H):
            for time_idx in range(T):
                timestamp = event_array[x, y, time_idx, 0]
                if timestamp > 0:
                    polarity = event_array[x, y, time_idx, 1]
                    events[count] = (x, y, timestamp, polarity)
                    count += 1

    return events[:count]


def hot_pixel_filter(events, x_dim, y_dim, threshold=10000):
    """Filter out events from oversensitive pixels"""
    # event counts per pixel
    x_coords = events['x']
    y_coords = events['y']

    # 2d histogram of events, helps identify hot pixels
    counts = np.zeros((x_dim, y_dim), dtype=np.int32)
    np.add.at(counts, (x_coords, y_coords), 1)

    # create mask for valid pixels
    valid_mask = counts[x_coords, y_coords] <= threshold
    return events[valid_mask]


def mask_region_filter(events, x_limits, y_limits):
    """Remove events in specified rectangular region"""
    mask = ~((events['x'] >= x_limits[0]) &
             (events['x'] <= x_limits[1]) &
             (events['y'] >= y_limits[0]) &
             (events['y'] <= y_limits[1]))
    return events[mask]


def normalize_image_3sigma(event_counts):
    """Normalize event counts using 3-sigma normalization"""
    non_zero = event_counts[event_counts > 0]
    if len(non_zero) == 0:
        return np.zeros_like(event_counts, dtype=np.uint8)  # return a zero array with same shape as original

    mean = non_zero.mean()
    std = non_zero.std()
    if std < 0.1 / 255:
        std = 0.1 / 255

    normalized = np.zeros_like(event_counts, dtype=np.float32)  # not sure if this should be np.float64 or not,
    # I remember that one issue we had
    mask = event_counts > 0
    normalized[mask] = (event_counts[mask] - mean) / (3 * std) + 0.5
    normalized = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    return normalized


def process_events_to_frames(events, events_per_frame=7500, resolution=(346, 260)):  # davis346
    """Process events into normalized frames with fixed event count"""
    # sort events chronologically, this should already be done but redundancy is good
    events = np.sort(events, order='t')

    frames = []
    total_events = len(events)

    for start in range(0, total_events, events_per_frame):
        end = start + events_per_frame
        if end > total_events:
            break

        # events for this frame
        frame_events = events[start:end]

        # make event count matrix
        event_counts = np.zeros(resolution, dtype=np.int32) # this might be np.int64, not sure
        x_coords = frame_events['x']
        y_coords = frame_events['y']
        np.add.at(event_counts, (x_coords, y_coords), 1)

        # normalize the events and store in frames
        normalized = normalize_image_3sigma(event_counts)
        frames.append(normalized)

    return frames


# Example usage
if __name__ == "__main__":
    # test event array
    W, H, T = 346, 260, 10
    event_array = np.zeros((W, H, T, 2), dtype=np.float32) # this might be np.float64, not sure

    event_array[100, 150, 0, 0] = 1000  # x=100, y=150, t=1000, pol=0
    event_array[100, 150, 1, 1] = 2000  # x=100, y=150, t=2000, pol=1
    event_array[200, 50, 0, 0] = 500  # x=200, y=50, t=500, pol=0

    # process events
    events = parse_events(event_array)
    events = hot_pixel_filter(events, W, H)
    events = mask_region_filter(events, (780, 810), (115, 145)) # can change rectangular filter
    frames = process_events_to_frames(events)

    # save to h5 file
    with h5py.File('processed_events.h5', 'w') as hf:
        for i, frame in enumerate(frames):
            hf.create_dataset(f'frame_{i:04d}', data=frame)
