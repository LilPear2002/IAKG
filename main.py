from config.configurator import configs
from trainer.trainer import init_seed, Trainer
from trainer.logger import Logger
from data_utils.data_handler_kg import DataHandlerKG
from models.iacd import IACD


def main():
    """Main training process"""
    # First Step: Initialize random seed
    init_seed()
    
    # Second Step: Create data handler and load data
    data_handler = DataHandlerKG()
    data_handler.load_data()

    # Third Step: Create model
    model = IACD(data_handler).to(configs['device'])

    # Fourth Step: Create logger
    logger = Logger()

    # Fifth Step: Create trainer
    trainer = Trainer(data_handler, logger)

    # Sixth Step: Training
    best_model = trainer.train(model)

    # Seventh Step: Test
    trainer.test(best_model)


if __name__ == '__main__':
    main()
