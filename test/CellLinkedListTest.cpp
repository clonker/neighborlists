#include <catch2/catch_test_macros.hpp>

#include <vector>

#include <neighborlists/Vec.h>
#include <neighborlists/CellLinkedList.h>
#include <random>


TEST_CASE("CellLinkedListSanity", "[cell_linked_list]") {
    using Vec1 = neighborlists::Vec<double, 1>;
    neighborlists::CellLinkedList<Vec1::dim, false, double> cll{{10.}, 1.};
    std::vector<Vec1> positions{
            {{-4.5}},
            {{-3.4}},
            {{3.}}  // only particles 0 and 1 are neighbors of one another
    };
    cll.update(begin(positions), end(positions), 2);

    std::vector<std::tuple<int, int>> pairs{};
    std::mutex m;
    cll.forEachParticlePair([&pairs, &m](auto i, auto j) {
        std::scoped_lock lock{m};
        pairs.push_back(std::make_tuple(i, j));
    }, 2);

    REQUIRE(std::find(begin(pairs), end(pairs), std::make_tuple(0, 1)) != end(pairs));
    REQUIRE(std::find(begin(pairs), end(pairs), std::make_tuple(1, 0)) != end(pairs));
}

TEST_CASE("CellLinkedListAgainstReference", "[cell_linked_list]") {
    std::mt19937 generator{33};
    using Vec3 = neighborlists::Vec<double, 3>;
    Vec3 boxSize{10, 10, 10};

    std::size_t nParticles = 1000;
    std::vector<Vec3> positions(nParticles);
    {
        for (std::size_t d = 0; d < Vec3::dim; ++d) {
            std::uniform_real_distribution<Vec3::value_type> dist{-boxSize[d] / 2, boxSize[d] / 2};
            for (std::size_t i = 0; i < nParticles; ++i) {
                positions[i][d] = dist(generator);
            }
        }
    }

    neighborlists::CellLinkedList<Vec3::dim, false, Vec3::value_type> cll{boxSize.data, 1.};
    cll.update(begin(positions), end(positions), 3);
    std::vector<std::tuple<int, int>> pairs{};
    {
        std::mutex m;
        cll.forEachParticlePair([&pairs, &m](auto i, auto j) {
            std::scoped_lock lock{m};
            pairs.push_back(std::make_tuple(i, j));
        }, 2);
    }

    for (auto i = 0U; i < nParticles; ++i) {
        for (auto j = i + 1; j < nParticles; ++j) {
            if ((positions[i] - positions[j]).normSquared() < 1) {
                REQUIRE(std::find(begin(pairs), end(pairs), std::make_tuple(i, j)) != end(pairs));
                REQUIRE(std::find(begin(pairs), end(pairs), std::make_tuple(j, i)) != end(pairs));
            }
        }
    }
}
