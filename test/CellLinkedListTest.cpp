#include <catch2/catch_test_macros.hpp>

#include <vector>

#include <neighborlists/Vec.h>
#include <neighborlists/CellLinkedList.h>


TEST_CASE("CellLinkedListSanity", "[cell_linked_list]") {
    using Vec1 = neighborlists::Vec<double, 1>;
    neighborlists::CellLinkedList<Vec1::dim, false, double> cll{{10.}, 1.};
    std::vector<Vec1> positions {
            {{-4.5}}, {{-3.4}}, {{3.}}  // only particles 0 and 1 are neighbors of one another
    };
    cll.update(begin(positions), end(positions), 2);

    std::vector<std::tuple<int, int>> pairs {};
    std::mutex m;
    cll.forEachParticlePair([&pairs, &m](auto i, auto j) {
        std::scoped_lock lock {m};
        pairs.push_back(std::make_tuple(i, j));
    }, 2);

    REQUIRE(std::find(begin(pairs), end(pairs), std::make_tuple(0, 1)) != end(pairs));
    REQUIRE(std::find(begin(pairs), end(pairs), std::make_tuple(1, 0)) != end(pairs));
}
