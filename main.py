import argparse
from generator import ImageGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Image Generation from Text Prompts")
    parser.add_argument("--learning_rate", type=float, default=0.5, help="Learning rate for optimization")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--noise_factor", type=float, default=0.22, help="Noise factor")
    parser.add_argument("--total_iter", type=int, default=400, help="Total number of iterations")
    parser.add_argument("--size1", type=int, default=450, help="Image height")
    parser.add_argument("--size2", type=int, default=450, help="Image width")
    parser.add_argument("--taming_config_path", type=str, required=True, help="Path to taming transformer config")
    parser.add_argument("--taming_checkpoint_path", type=str, required=True,
                        help="Path to taming transformer checkpoint")
    parser.add_argument("--include", nargs='+', required=True, help="List of text prompts to include")
    parser.add_argument("--exclude", type=str, default="", help="Text prompt to exclude")
    parser.add_argument("--extras", type=str, default="", help="Extra text prompt")
    parser.add_argument("--output_video", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--w1", type=float, default=1.0, help="Weight for include prompts")
    parser.add_argument("--w2", type=float, default=1.0, help="Weight for extras prompt")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for main loss")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight for penalize loss")
    parser.add_argument("--show_step", type=int, default=10, help="Step interval for showing results")
    parser.add_argument("--num_crops", type=int, default=32, help="Number of crops for CLIP processing")
    return parser.parse_args()


def main():
    args = parse_args()
    generator = ImageGenerator(args)
    res_img, res_z = generator.generate()
    interp_result_img_list = generator.interpolate(res_z, [5] * len(res_z))
    generator.create_video(interp_result_img_list, args.output_video)


if __name__ == "__main__":
    main()
