from dataclasses import dataclass

@dataclass
class dataIngestionArtifact:
    trained_file_path: str
    test_file_path: str