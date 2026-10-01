#include "strata/program/expert_usage.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
int main() {
    const auto dir = std::filesystem::temp_directory_path() /
        ("strata-usage-test-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()));
    {
        strata::program::ExpertUsage usage;
        if (!usage.start(dir.string(), 2, 3)) throw std::runtime_error("start failed");
        const int32_t ids[] = {2, 1, 2, -1, 9, 0};
        usage.record(1, ids, 2, 3);
        usage.record(-1, ids, 1, 3);
        usage.publish();
        usage.record(0, ids, 1, 3);
    } // destructor drains the latest snapshot
    int files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() != ".tsv") throw std::runtime_error("leftover temporary file");
        std::ifstream in(entry.path());
        std::string contents((std::istreambuf_iterator<char>(in)), {});
        if (contents != "# strata-expert-usage-v1 2 3\n0\t1\t1\n0\t2\t2\n1\t0\t1\n1\t1\t1\n1\t2\t2\n")
            throw std::runtime_error("incorrect cumulative counts");
        ++files;
    }
    std::filesystem::remove_all(dir);
    if (files != 1) throw std::runtime_error("expected one bounded snapshot");
}
