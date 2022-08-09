#include <catch2/catch_test_macros.hpp>

#include <vector>

#include <neighborlists/Vec.h>
#include <neighborlists/CellLinkedList.h>
#include "spdlog/spdlog.h"

TEST_CASE("CellLinkedListSanity", "[cell_linked_list]") {
    using Vec3 = neighborlists::Vec<double, 3>;
    std::vector<Vec3> positions;
    for(int i = 0; i < 100; ++i) {
        positions.push_back({{i / 10. - 5., i/10. - 5., i / 10. - 5.}});
    }

    neighborlists::CellLinkedList<3, false, double> cll {{10., 10., 10.}, .5};
    cll.update(begin(positions), end(positions), 2);

    std::atomic<std::uint32_t> pairs;
    cll.forEachParticlePair([&positions, &pairs](auto i, auto j) {
        if((positions[i] - positions[j]).norm() < .5) {
            ++pairs;
        }
    }, 1);

    std::uint32_t referencePairs {};
    for(std::size_t i = 0; i < positions.size(); ++i) {
        for(std::size_t j = 0; j < positions.size(); ++j) {
            if(i != j && (positions[i] - positions[j]).norm() < .5) {
                ++referencePairs;
            }
        }
    }

    REQUIRE(pairs.load() == referencePairs);
}
