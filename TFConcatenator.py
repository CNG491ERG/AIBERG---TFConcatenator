import tensorflow as tf
import os


def get_last_step(log_path):
    last_step = 0
    for record in tf.data.TFRecordDataset(log_path):
        event = tf.compat.v1.Event.FromString(record.numpy())
        if event.step > last_step:
            last_step = event.step
    return last_step


def adjust_and_concatenate_tf_logs(input_paths, output_path):
    offsets = [0]
    last_step = 0

    # Calculate the offsets for each file
    for i in range(1, len(input_paths)):
        last_step += get_last_step(input_paths[i - 1])
        offsets.append(last_step)

    # Write adjusted event logs
    with tf.io.TFRecordWriter(output_path) as writer:
        for input_path, offset in zip(input_paths, offsets):
            for raw_record in tf.data.TFRecordDataset(input_path):
                event = tf.compat.v1.Event.FromString(raw_record.numpy())
                # Adjust the step count
                if event.HasField('summary'):
                    for value in event.summary.value:
                        if value.HasField('simple_value'):
                            event.step += offset
                writer.write(event.SerializeToString())


def process_agent_directories(base_dir):
    agents = ['Boss', 'Player']
    for agent in agents:
        agent_dir = os.path.join(base_dir, agent)
        if os.path.exists(agent_dir):
            input_paths = get_session_paths(agent_dir)
            output_dir = os.path.join(base_dir, 'Results', agent)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, 'events.out.tfevents.concatenated')
            adjust_and_concatenate_tf_logs(input_paths, output_path)
        else:
            print(f"No directory found for agent {agent}")


def get_session_paths(agent_dir):
    # Assumes that session directories are named with integers in increasing order
    sessions = [d for d in os.listdir(agent_dir) if os.path.isdir(os.path.join(agent_dir, d))]
    sessions.sort(key=lambda x: int(x))  # Sort directories numerically
    session_paths = [os.path.join(agent_dir, s, f) for s in sessions for f in os.listdir(os.path.join(agent_dir, s)) if
                     'tfevents' in f]
    return session_paths


def main():
    base_dir = os.getcwd()
    process_agent_directories(base_dir)


if __name__ == '__main__':
    main()
