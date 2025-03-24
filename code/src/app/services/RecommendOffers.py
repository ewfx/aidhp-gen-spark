class RecommendOffers:

    def __init__(self, input_individual, offers_data_individual):
        self.input_individual = input_individual
        self.offers_data_individual = offers_data_individual

    def generate_merchant_offer(self):
        # Process the data
        return self.offers_data_individual.merchant_offers

    def generate_bank_offer(self):
        # Process the data
        return self.offers_data_individual.bank_offers
