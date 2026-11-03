import argparse
from pathlib import Path
from src.train import train
from src.evaluate import evaluate
from src.visualization import plot_results, create_demo_videos, create_worst_case_video
from src.export_results import generate_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['quick', 'full'], default='quick')
    p.add_argument('--output', required=True)
    p.add_argument('--videos', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--mirror')
    a = p.parse_args()
    root = Path(a.output).resolve()
    training, evaluation = root / 'training', root / 'evaluation'
    checkpoint = training / 'checkpoints/training_checkpoint.pt' if a.resume else None
    train(a.mode, training, resume=checkpoint, mirror=a.mirror)
    evaluate(training / 'checkpoints/best_model.pt', evaluation, mode=a.mode, resume=a.resume)
    plot_results(training, evaluation)
    report = generate_report(training, evaluation)
    if a.videos:
        create_demo_videos(training / 'checkpoints/best_model.pt', evaluation)
        create_worst_case_video(training / 'checkpoints/best_model.pt', evaluation)
    print(report)


if __name__ == '__main__':
    main()
