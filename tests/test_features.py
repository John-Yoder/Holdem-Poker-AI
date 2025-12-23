from poker.helpers.features import extract_features

def test_overpair_detection():
    feats = extract_features(["Ah","Ad"], ["Ks","7d","2c"])
    assert feats["overpair"] is True

def test_top_pair_detection():
    feats = extract_features(["Kc","9d"], ["Ks","7d","2c"])
    assert feats["top_pair"] is True
