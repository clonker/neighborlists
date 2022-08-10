#include <vector>
#include <random>

#include <spdlog/spdlog.h>

#include <neighborlists/Vec.h>
#include <neighborlists/CellLinkedList.h>

template<typename Generator = std::mt19937>
Generator randomlySeededGenerator() {
    std::random_device r;
    std::random_device::result_type threadId = std::hash<std::thread::id>()(std::this_thread::get_id());
    std::random_device::result_type clck = clock();
    std::seed_seq seed{threadId, r(), r(), r(), clck, r(), r(), r(), r(), r()};
    return Generator(seed);
}

int main() {
    spdlog::set_level(spdlog::level::debug);

    std::uint32_t nJobs{8};

    using Vec3 = neighborlists::Vec<double, 3>;
    Vec3::value_type cutoff = 0.5;
    Vec3 boxSize{10., 10., 10.};

    std::size_t nParticles = 20000;
    std::vector<Vec3> positions(nParticles);
    {
        auto generator = randomlySeededGenerator();
        for (std::size_t d = 0; d < Vec3::dim; ++d) {
            std::uniform_real_distribution<Vec3::value_type> dist{-boxSize[d] / 2, boxSize[d] / 2};
            for (std::size_t i = 0; i < nParticles; ++i) {
                positions[i][d] = dist(generator);
            }
        }
    }

    {
        std::atomic<std::uint32_t> pairs;

        neighborlists::CellLinkedList<Vec3::dim, false, Vec3::value_type> cll{boxSize.data, cutoff};
        auto start = std::chrono::high_resolution_clock::now();
        cll.update(begin(positions), end(positions), nJobs);
        cll.forEachParticlePair([&positions, &pairs, cutoff](auto i, auto j) {
            if ((positions[i] - positions[j]).norm() < cutoff) {
                ++pairs;
            }
        }, nJobs);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        spdlog::debug("Elapsed for CLL {}ms = {} * {}ms ({} pairs)", elapsed.count(), nJobs,
                      elapsed.count() / static_cast<float>(nJobs), pairs.load());
    }

    {
        std::uint32_t referencePairs{};
        auto start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < positions.size(); ++i) {
            for (std::size_t j = 0; j < positions.size(); ++j) {
                if (i != j && (positions[i] - positions[j]).norm() < cutoff) {
                    ++referencePairs;
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        spdlog::debug("Elapsed for reference {}ms = {} * {}ms ({} pairs)", elapsed.count(), nJobs,
                      elapsed.count() / static_cast<float>(nJobs), referencePairs);
    }

    {
        std::uint32_t referencePairs{};
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel default(none) shared(positions, referencePairs) firstprivate(cutoff)
        {
            std::uint32_t referencePairsLocal {};
            #pragma omp for collapse(2)
            for (std::size_t i = 0; i < positions.size(); ++i) {
                for (std::size_t j = 0; j < positions.size(); ++j) {
                    if (i != j && (positions[i] - positions[j]).norm() < cutoff) {
                        ++referencePairsLocal;
                    }
                }
            }

            #pragma omp critical
            referencePairs += referencePairsLocal;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        spdlog::debug("Elapsed for openmp loop {}ms = {} * {}ms ({} pairs)", elapsed.count(), nJobs,
                      elapsed.count() / static_cast<float>(nJobs), referencePairs);
    }

    return 0;
}
