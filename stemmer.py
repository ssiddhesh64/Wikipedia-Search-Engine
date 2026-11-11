"""
stemmer.py

Pure-Python implementation of the classic Martin Porter Stemming Algorithm (1980).
Zero external dependencies. Reduces morphological variants to their common root
(e.g., 'algorithms' -> 'algorithm', 'searching' -> 'search', 'computational' -> 'comput').
"""


class PorterStemmer:
    def __init__(self):
        self.b = ""  # buffer for word to be stemmed
        self.k = 0
        self.k0 = 0
        self.j = 0  # a general offset into the string

    def _is_consonant(self, i):
        letter = self.b[i]
        if letter in ("a", "e", "i", "o", "u"):
            return False
        if letter == "y":
            if i == self.k0:
                return True
            else:
                return not self._is_consonant(i - 1)
        return True

    def _measure(self):
        """Measures the number of consonant sequences between k0 and j."""
        n = 0
        i = self.k0
        while True:
            if i > self.j:
                return n
            if not self._is_consonant(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return n
                if self._is_consonant(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self._is_consonant(i):
                    break
                i += 1
            i += 1

    def _vowel_in_stem(self):
        for i in range(self.k0, self.j + 1):
            if not self._is_consonant(i):
                return True
        return False

    def _double_consonant(self, i):
        if i < self.k0 + 1:
            return False
        if self.b[i] != self.b[i - 1]:
            return False
        return self._is_consonant(i)

    def _cvc(self, i):
        if i < self.k0 + 2 or not self._is_consonant(i) or self._is_consonant(i - 1) or not self._is_consonant(i - 2):
            return False
        ch = self.b[i]
        if ch in ("w", "x", "y"):
            return False
        return True

    def _ends(self, s):
        length = len(s)
        o = self.k - length + 1
        if o < self.k0:
            return False
        for i in range(length):
            if self.b[o + i] != s[i]:
                return False
        self.j = self.k - length
        return True

    def _set_to(self, s):
        length = len(s)
        o = self.j + 1
        for i in range(length):
            self.b[o + i] = s[i]
        self.k = self.j + length

    def _replace(self, s):
        if self._measure() > 0:
            self._set_to(s)

    def _step1ab(self):
        if self.b[self.k] == "s":
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._set_to("i")
            elif self.b[self.k - 1] != "s":
                self.k -= 1
        if self._ends("eed"):
            if self._measure() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowel_in_stem():
            self.k = self.j
            if self._ends("at"):
                self._set_to("ate")
            elif self._ends("bl"):
                self._set_to("ble")
            elif self._ends("iz"):
                self._set_to("ize")
            elif self._double_consonant(self.k):
                self.k -= 1
                ch = self.b[self.k]
                if ch in ("l", "s", "z"):
                    self.k += 1
            elif self._measure() == 1 and self._cvc(self.k):
                self._set_to("e")

    def _step1c(self):
        if self._ends("y") and self._vowel_in_stem():
            self.b[self.k] = "i"

    def _step2(self):
        if self.k <= self.k0:
            return
        ch = self.b[self.k - 1]
        if ch == "a":
            if self._ends("ational"):
                self._replace("ate")
            elif self._ends("tional"):
                self._replace("tion")
        elif ch == "c":
            if self._ends("enci"):
                self._replace("ence")
            elif self._ends("anci"):
                self._replace("ance")
        elif ch == "e":
            if self._ends("izer"):
                self._replace("ize")
        elif ch == "l":
            if self._ends("bli"):
                self._replace("ble")
            elif self._ends("alli"):
                self._replace("al")
            elif self._ends("entli"):
                self._replace("ent")
            elif self._ends("eli"):
                self._replace("e")
            elif self._ends("ousli"):
                self._replace("ous")
        elif ch == "o":
            if self._ends("ization"):
                self._replace("ize")
            elif self._ends("ation"):
                self._replace("ate")
            elif self._ends("ator"):
                self._replace("ate")
        elif ch == "s":
            if self._ends("alism"):
                self._replace("al")
            elif self._ends("iveness"):
                self._replace("ive")
            elif self._ends("fulness"):
                self._replace("ful")
            elif self._ends("ousness"):
                self._replace("ous")
        elif ch == "t":
            if self._ends("aliti"):
                self._replace("al")
            elif self._ends("iviti"):
                self._replace("ive")
            elif self._ends("biliti"):
                self._replace("ble")

    def _step3(self):
        if self.k <= self.k0:
            return
        ch = self.b[self.k]
        if ch == "e":
            if self._ends("icate"):
                self._replace("ic")
            elif self._ends("ative"):
                self._replace("")
            elif self._ends("alize"):
                self._replace("al")
        elif ch == "i":
            if self._ends("iciti"):
                self._replace("ic")
        elif ch == "l":
            if self._ends("ical"):
                self._replace("ic")
            elif self._ends("ful"):
                self._replace("")
        elif ch == "s":
            if self._ends("ness"):
                self._replace("")

    def _step4(self):
        if self.k <= self.k0 + 1:
            return
        ch = self.b[self.k - 1]
        if ch == "a":
            if self._ends("al"):
                pass
            else:
                return
        elif ch == "c":
            if self._ends("ance") or self._ends("ence"):
                pass
            else:
                return
        elif ch == "e":
            if self._ends("er"):
                pass
            else:
                return
        elif ch == "i":
            if self._ends("ic"):
                pass
            else:
                return
        elif ch == "l":
            if self._ends("able") or self._ends("ible"):
                pass
            else:
                return
        elif ch == "n":
            if self._ends("ant") or self._ends("ement") or self._ends("ment") or self._ends("ent"):
                pass
            else:
                return
        elif ch == "o":
            if self._ends("ion") and self.j >= 0 and self.b[self.j] in ("s", "t"):
                pass
            elif self._ends("ou"):
                pass
            else:
                return
        elif ch == "s":
            if self._ends("ism"):
                pass
            else:
                return
        elif ch == "t":
            if self._ends("ate") or self._ends("iti"):
                pass
            else:
                return
        elif ch == "u":
            if self._ends("ous"):
                pass
            else:
                return
        elif ch == "v":
            if self._ends("ive"):
                pass
            else:
                return
        elif ch == "z":
            if self._ends("ize"):
                pass
            else:
                return
        else:
            return

        if self._measure() > 1:
            self.k = self.j

    def _step5(self):
        self.j = self.k
        if self.b[self.k] == "e":
            a = self._measure()
            if a > 1 or (a == 1 and not self._cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == "l" and self._double_consonant(self.k) and self._measure() > 1:
            self.k -= 1

    def stem(self, word: str) -> str:
        word = word.lower()
        if len(word) <= 2:
            return word
        self.b = list(word)
        self.k = len(word) - 1
        self.k0 = 0
        self._step1ab()
        self._step1c()
        self._step2()
        self._step3()
        self._step4()
        self._step5()
        return "".join(self.b[: self.k + 1])


_default_stemmer = PorterStemmer()


def stem(word: str) -> str:
    """Convenience helper to stem a single word."""
    return _default_stemmer.stem(word)

