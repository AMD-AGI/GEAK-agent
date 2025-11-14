import re
import argparse

def parse_log_file(log_path):
    results = []
    with open(log_path) as f:
        content = f.read()

    # Split on ==== <path> ====, capturing file path and base name
    blocks = re.split(r'==== (.+?/([^/]+)/run\.log) ====', content)[1:]
    for i in range(0, len(blocks), 3):
        full_path, file_name, block_text = blocks[i], blocks[i+1], blocks[i+2]
        # Find all checkpoint blocks
        matches = re.finditer(r'checkpoint (\d+):.*?(?=checkpoint |\Z)', block_text, flags=re.DOTALL)
        for match in matches:
            checkpoint = int(match.group(1))
            island_block = match.group(0)
            best_vals = re.findall(r'best=([\d.]+)', island_block)
            if best_vals:
                best_val = max(map(float, best_vals))
                results.append((file_name, checkpoint, best_val))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse run.log for best island fitness per checkpoint.")
    parser.add_argument("logfile", type=str, help="Path to the combined log file")
    args = parser.parse_args()
    print(f"kernel,iteration,best")
    for file_name, checkpoint, best in parse_log_file(args.logfile):
        print(f"{file_name},{checkpoint},{best}")
