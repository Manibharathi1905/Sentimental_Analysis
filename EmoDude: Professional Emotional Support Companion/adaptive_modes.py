class AdaptiveModes:
    def __init__(self):
        self.emotional_support_mode = True
        self.motivational_mode = False

    def toggle_mode(self, mode):
        """Toggle between Emotional Support and Motivational modes."""
        if mode == "emotional_support":
            self.emotional_support_mode = True
            self.motivational_mode = False
        elif mode == "motivational":
            self.emotional_support_mode = False
            self.motivational_mode = True

    def get_active_mode(self):
        """Return the currently active mode."""
        if self.emotional_support_mode:
            return "Emotional Support Mode"
        elif self.motivational_mode:
            return "Motivational Mode"

# Global instance
adaptive_modes = AdaptiveModes()