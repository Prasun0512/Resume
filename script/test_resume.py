"""Check the published PDF against its public source without external services."""
import json
from pathlib import Path
import unittest

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def normalized(text):
    return " ".join(text.split()).casefold()


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads((ROOT / "resume/profile.json").read_text(encoding="utf-8"))
        cls.reader = PdfReader(ROOT / "Prasun_Kumar_AI_ML_Architect_Resume.pdf")
        cls.text = normalized(" ".join(page.extract_text() for page in cls.reader.pages))

    def test_two_selectable_text_pages(self):
        self.assertEqual(len(self.reader.pages), 2)
        for page in self.reader.pages:
            self.assertGreater(len(page.extract_text().split()), 250)

    def test_contact_and_summary_match(self):
        for key in ("name", "headline", "current_role", "contact", "summary"):
            self.assertIn(normalized(self.source[key]), self.text)

    def test_employment_is_current_and_complete(self):
        jobs = self.source["employment"]
        self.assertEqual(sum("Present" in job["dates"] for job in jobs), 1)
        self.assertIn("IBM", jobs[0]["company"])
        self.assertIn("3 November 2026", jobs[0]["bullets"][0])
        for job in jobs:
            for value in [job["role"], job["company"], job["dates"], *job["bullets"]]:
                self.assertIn(normalized(value), self.text)

    def test_projects_remain_separate_and_complete(self):
        boundary = self.text.index("independent github projects")
        for project in self.source["projects"]:
            self.assertLess(self.text.index(normalized(project["name"])), boundary)
            for bullet in project["bullets"]:
                self.assertIn(normalized(bullet), self.text)
        for project in self.source["independent"]:
            self.assertGreater(self.text.index(normalized(project["name"])), boundary)
            self.assertIn(normalized(project["text"]), self.text)

    def test_links_are_present(self):
        urls = set()
        for page in self.reader.pages:
            for ref in page.get("/Annots", []):
                action = ref.get_object().get("/A", {})
                if "/URI" in action:
                    urls.add(str(action["/URI"]))
        expected = {entry["url"] for entry in self.source["links"]}
        expected.add(self.source["credential"])
        expected.update("https://github.com/Prasun0512/" + item["repo"]
                        for item in self.source["projects"] + self.source["independent"] if item["repo"])
        self.assertTrue(expected.issubset(urls))

    def test_private_application_details_are_excluded(self):
        for term in ("current ctc", "target ctc", "salary", "lpa", "notice period", "verification required"):
            self.assertNotIn(term, self.text)


if __name__ == "__main__":
    unittest.main()
