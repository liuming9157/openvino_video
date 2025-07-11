from pipeline import OVPipeline


parser.add_argument(
    "--prompt",
    type=str,
    help="prompt for video generation",
    default="",
)


def main():
    output = OVPipeline.generate_video(
        prompt=args.prompt,
    )
    print(output)


if __name__ == "__main__":
    main()
