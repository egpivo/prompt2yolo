from prompt2yolo.configs import ImageGeneratorConfig
from prompt2yolo.data.data_generation.image_generators.pixart_generator import (
    PixartGenerator,
)
from prompt2yolo.data.data_generation.image_generators.realtek_generator import (
    RealtekGenerator,
)
from prompt2yolo.utils.s3_handler import S3Handler


class GeneratorFactory:
    @staticmethod
    def get_generator(s3_handler: S3Handler, config: ImageGeneratorConfig):
        generators = {"pixart": PixartGenerator, "realtek": RealtekGenerator}
        generator_type = config.generator
        if generator_type.lower() in generators:
            return generators[generator_type.lower()](s3_handler, config)
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
