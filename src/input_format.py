class InputExample(object):
    def __init__(self, guk, sent, idx, answer=None, mentions=None, img_path=None):
        self.guk = guk  # The unique id of each example, generally composed of mode-key
        self.sent = sent  # Sample text information
        self.img_id = idx  # The original id of the sample, used to retrieve the image
        self.answer = answer  # The answer information corresponding to the sample, that is, the id of the database instance
        self.mentions = mentions  # Reference information in the sample
        self.img_path = img_path


class InputFeatures:
    def __init__(self, answer_id, mentions, text_feature, mention_feature, total_feature=None,
                 segement_feature=None, profile_feature=None, identity_feature=None):
        self.answer_id = answer_id
        self.mentions = mentions
        self.text_feature = text_feature
        self.mention_feature = mention_feature
        self.total_feature = total_feature
        self.segement_feature = segement_feature
        self.profile_feature = profile_feature
        self.identity_feature = identity_feature
 