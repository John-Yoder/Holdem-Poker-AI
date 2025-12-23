from poker.helpers.texture import board_texture

def test_paired_board():
    tex = board_texture(["Ah","Ad","7s"])
    assert tex["paired_board"] is True
    assert tex["ace_high_board"] is True

def test_flush_draw_available_on_four_suit_board():
    tex = board_texture(["Ah","Kh","2h","7h"])
    assert tex["flush_draw_available"] is True
    assert tex["flush_available"] is False

def test_straight_draw_available():
    tex = board_texture(["7s","8d","9c","Jh"])
    assert tex["straight_draw_available"] is True
