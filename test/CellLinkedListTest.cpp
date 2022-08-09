#include <catch2/catch_test_macros.hpp>

#include <vector>

#include <neighborlists/Vec.h>
#include <neighborlists/CellLinkedList.h>
#include <random>
#include "spdlog/spdlog.h"

template<typename Generator = std::mt19937>
Generator randomlySeededGenerator() {
    std::random_device r;
    std::random_device::result_type threadId = std::hash<std::thread::id>()(std::this_thread::get_id());
    std::random_device::result_type clck = clock();
    std::seed_seq seed{threadId, r(), r(), r(), clck, r(), r(), r(), r(), r()};
    return Generator(seed);
}

TEST_CASE("CellLinkedListSanity", "[cell_linked_list]") {
    using Vec3 = neighborlists::Vec<double, 3>;

    std::size_t nParticles = 20000;
    std::vector<Vec3> positions (nParticles);
    {
        auto generator = randomlySeededGenerator();
        for (std::size_t i = 0; i < nParticles; ++i) {
            for (std::size_t d = 0; d < 3; ++d) {
                std::uniform_real_distribution<float> dist{-5., 5.};
                positions[i][d] = dist(generator);
            }
        }
    }

    std::atomic<std::uint32_t> pairs;
    {
        auto start = std::chrono::high_resolution_clock::now();
        neighborlists::CellLinkedList<3, false, double> cll{{10., 10., 10.}, .5};
        cll.update(begin(positions), end(positions), 8);
        cll.forEachParticlePair([&positions, &pairs](auto i, auto j) {
            if ((positions[i] - positions[j]).norm() < .5) {
                ++pairs;
            }
        }, 8);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        spdlog::error("Elapsed for CLL {}ms", elapsed.count());
    }

    std::uint32_t referencePairs{};
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < positions.size(); ++i) {
            for (std::size_t j = 0; j < positions.size(); ++j) {
                if (i != j && (positions[i] - positions[j]).norm() < .5) {
                    ++referencePairs;
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        spdlog::error("Elapsed for reference {}ms", elapsed.count());
    }

    REQUIRE(pairs.load() == referencePairs);
}
