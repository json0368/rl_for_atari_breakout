"""Show a saved policy in a window, or record MP4 gameplay."""
from .evaluate import parser, run

def main():
    p = parser()
    p.description = __doc__
    p.set_defaults(episodes=3)
    args = p.parse_args()
    try:
        run(args, human=args.video_dir is None)
    except KeyboardInterrupt:
        print("Demo stopped.")

if __name__ == "__main__":
    main()
