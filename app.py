from pipeline import OVPipeline


parser.add_argument(
    "--prompt",
    type=str,
    help="prompt for video generation",
    default="",
)


def main():
    prompt = f"""
    请将下面的古诗翻译成现代通俗易懂的语言,细节尽可能丰富，能让人在大脑中形成对应的画面。
    1. 只保留画面描述的内容，不要包含任何其他信息。
    2. 古诗内容如下：
    {args.prompt}
    """
    output = OVPipeline.generate_video(
        prompt=prompt,
    )
    print(output)


if __name__ == "__main__":
    main()
