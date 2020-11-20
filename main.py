import logging
import ray
import pprint

from argparser import parser


def run():
    # choose and load model

    # training loop

        # sample from the episodes
            # look how dreamer samples from the disk

        # eval after every log_interval

        # video log after every video_log_interval

    # save model progress repeatedly

    pass


if __name__ == "__main__":
    args = parser.parse_args()
    run(args.config_file, args.override_config_file)
