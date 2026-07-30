from openclaw import OpenClaw
from chroma_query import chroma_query_executor as cqe

def main():
    app = OpenClaw()

    cq = cqe().get_collection_count()

    app.add_skill(cq)

    app.run()

if __name__ == "__main__":
    main()