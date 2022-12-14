#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <array>
#include <thread>
#include <future>

#include "Index.h"
#include "CopyableAtomic.h"

namespace neighborlists {

/**
 * Struct computing and storing the neighboring cells in a CLL.
 *
 * @tparam Index Type of index that is used in the CLL.
 * @tparam periodic Whether periodic boundary conditions apply.
 */
template<typename Index, bool periodic>
struct CellAdjacency {
    static_assert(std::is_signed_v<typename Index::value_type>, "Needs index to be signed!");

    /**
     * Check whether a multi-index ijk is within bounds of the index itself.
     *
     * @param index The index.
     * @param ijk A multi index for which it is checked, whether it is still within bounds of index.
     * @return true if none of the index dimensions are exceeded and none of the entries are negative, else false
     */
    bool withinBounds(const Index &index, const typename Index::GridDims &ijk) const {
        const bool nonNegative = std::all_of(begin(ijk), end(ijk), [](const auto &element) { return element >= 0; });
        bool nonExceeding = true;
        for (std::size_t dim = 0; dim < Index::Dims; ++dim) {
            nonExceeding &= ijk[dim] < index[dim];
        }
        return nonNegative && nonExceeding;
    }

    /**
     * Finds adjacent cell indices given a current cell (specified by ijk) in a certain dimensional direction and
     * stores those inside an adjacency vector.
     *
     * @param index The reference index, specifying the grid.
     * @param ijk The current cell.
     * @param dim Dimension axis to explore.
     * @param r Number of subdivisions.
     * @param adj Vector with adjacent cells.
     */
    void findEachAdjCell(const Index &index, const typename Index::GridDims &ijk, std::size_t dim, std::int32_t r,
                         std::vector<std::size_t> &adj) const {
        for (int ii = ijk[dim] - r; ii <= ijk[dim] + r; ++ii) {
            typename Index::GridDims cix{ijk};
            cix[dim] = ii;
            if constexpr (periodic) {
                cix[dim] = (cix[dim] % index[dim] + index[dim]) % index[dim];  // wrap around
            }

            if (withinBounds(index, cix)) {
                if (dim > 0) {
                    findEachAdjCell(index, cix, dim - 1, r, adj);
                } else {
                    adj.push_back(index.index(cix));
                }
            }
        }
    }

    CellAdjacency() = default;

    /**
     * Creates and populates new cell adjacency struct.
     *
     * @param index The grid index object.
     * @param radius Radius in which to check for neighbors (in terms of discrete steps around a reference cell).
     */
    CellAdjacency(const Index &index, const std::int32_t radius) {
        typename Index::GridDims nNeighbors{};
        for (std::size_t i = 0; i < Index::Dims; ++i) {
            nNeighbors[i] = std::min(index[i], static_cast<decltype(index[i])>(2 * radius + 1));
        }
        // number of neighbors plus the cell itself
        const auto nAdjacentCells = 1 + std::accumulate(begin(nNeighbors), end(nNeighbors), 1, std::multiplies<>());
        // number of neighbors plus cell itself plus padding to save how many neighbors actually
        cellNeighbors = neighborlists::Index<2>{std::array<typename Index::value_type, 2>{index.size(),
                                                                                          static_cast<typename Index::value_type>(
                                                                                                  1 + nAdjacentCells)}};
        cellNeighborsContent.resize(cellNeighbors.size());

        std::vector<std::size_t> adj;
        adj.reserve(1 + nAdjacentCells);
        for (std::size_t i = 0; i < index.size(); ++i) {
            adj.resize(0);
            const auto ijk = index.inverse(i);
            findEachAdjCell(index, ijk, Index::Dims - 1, radius, adj);
            std::sort(adj.begin(), adj.end());
            adj.erase(std::unique(std::begin(adj), std::end(adj)), std::end(adj));

            const auto begin = cellNeighbors(i, 0);
            cellNeighborsContent[begin] = adj.size();
            std::copy(adj.begin(), adj.end(), &cellNeighborsContent.at(begin + 1));
        }
    }

    CellAdjacency(const CellAdjacency &) = delete;

    CellAdjacency &operator=(const CellAdjacency &) = delete;

    CellAdjacency(CellAdjacency &&) noexcept = default;

    CellAdjacency &operator=(CellAdjacency &&) noexcept = default;

    ~CellAdjacency() = default;

    /**
     * The number of neighboring cells for specific (flat) cell index.
     *
     * @param cellIndex The cell index.
     * @return Number of neighbors.
     */
    [[nodiscard]] auto nNeighbors(auto cellIndex) const {
        return cellNeighborsContent[cellNeighbors(cellIndex, 0)];
    }

    /**
     * Random-access iterator denoting the begin of cell neighbors of cellIndex's cell.
     *
     * @param cellIndex The reference cell
     * @return iterator with begin
     */
    [[nodiscard]] auto cellsBegin(auto cellIndex) const {
        return begin(cellNeighborsContent) + cellNeighbors(cellIndex, 1);
    }

    /**
     * See cellsBegin.
     */
    [[nodiscard]] auto cellsEnd(auto cellIndex) const {
        return cellsBegin(cellIndex) + nNeighbors(cellIndex);
    }

    // index of size (n_cells x (1 + nAdjacentCells)), where the first element tells how many adj cells are stored
    neighborlists::Index<2> cellNeighbors{};
    // backing vector of _cellNeighbors index of size (n_cells x (1 + nAdjacentCells))
    std::vector<std::size_t> cellNeighborsContent{};
};

/**
 * A cell linked-list.
 *
 * @tparam DIM the dimension
 * @tparam periodic whether periodic boundary conditions apply
 * @tparam dtype the data type of positions (typically float/double)
 */
template<int DIM, bool periodic, typename dtype>
class CellLinkedList {
public:
    /**
     * Index type which is used to access cells as spatial (i, j, k,...) indices or flat (ravel/unravel).
     */
    using Index = neighborlists::Index<DIM, std::array<std::int32_t, DIM>>;

    /**
     * Creates a new CLL based on a grid size (which assumed to result in an origin-centered space), an interaction
     * radius which is used to determine the amount of subdivision and a number of subdivides to further fine-grain
     * the sphere approximation.
     *
     * @param gridSize array of dtype describing the extent of space
     * @param interactionRadius maximum radius under which particles can interact
     * @param nSubdivides amount of fine-graining
     */
    CellLinkedList(std::array<dtype, DIM> gridSize, dtype interactionRadius, int nSubdivides = 2)
        : _gridSize(gridSize) {
        // determine the number of cells per axis
        std::array<typename Index::value_type, DIM> nCells;
        for (int i = 0; i < DIM; ++i) {
            _cellSize[i] = interactionRadius / nSubdivides;
            if (gridSize[i] <= 0) {
                throw std::invalid_argument("grid sizes must be positive.");
            }
            nCells[i] = gridSize[i] / _cellSize[i];
        }
        // create index for raveling and unravling operations
        _index = Index(nCells);
        // initialize head to reflect the total number of cells
        head.resize(_index.size());
        // compute adjacencies among cells and store in arrays
        _adjacency = CellAdjacency<Index, periodic>{_index, nSubdivides};
    }

    ~CellLinkedList() = default;

    CellLinkedList(const CellLinkedList &) = delete;
    CellLinkedList &operator=(const CellLinkedList &) = delete;
    CellLinkedList(CellLinkedList &&) noexcept = default;
    CellLinkedList &operator=(CellLinkedList &&) noexcept = default;

    /**
     * Clears the currently stored particle indices and (re)computes the cell linked-list structure.
     *
     * @tparam Iterator Type of random-access iterator.
     * @param begin begin of the data structure containing vectors
     * @param end end of the data structure containing vectors
     * @param nJobs number of processes to use, threads are automatically joined
     */
    template<std::random_access_iterator Iterator>
    void update(Iterator begin, Iterator end, std::uint32_t nJobs) {
        // add artificial empty particle so that all entries can be unsigned
        list.resize(std::distance(begin, end) + 1);
        // reset list
        std::fill(std::begin(list), std::end(list), 0);
        // reset head
        std::fill(std::begin(head), std::end(head), CopyableAtomic<std::size_t>());

        // operation which updates the cell linked-list by one individual particle based on its position
        const auto updateOp = [this](std::size_t particleId, const auto &pos) {
            const auto boxId = positionToCellIndex(&pos.data[0]);

            // CAS
            auto &atomic = *head.at(boxId);
            auto currentHead = atomic.load();
            while (!atomic.compare_exchange_weak(currentHead, particleId + 1)) {}
            list[particleId + 1] = currentHead;
        };

        // depending on whether == 1 or nJobs > 1, either just iterate or use threads
        if (nJobs == 1) {
            std::size_t id = 0;
            for (std::random_access_iterator auto it = begin; it != end; ++it, ++id) {
                updateOp(id, *it);
            }
        } else {
            std::size_t grainSize = std::distance(begin, end) / nJobs;
            std::vector<std::jthread> jobs;
            for (int i = 0; i < nJobs - 1; ++i) {
                auto next = std::min(begin + grainSize, end);
                if (begin != next) {
                    std::size_t idStart = i * grainSize;
                    jobs.emplace_back([&updateOp, idStart, begin, next]() {
                        auto id = idStart;
                        for (std::random_access_iterator auto it = begin; it != next; ++it, ++id) {
                            updateOp(id, *it);
                        }
                    });
                }
                begin = next;
            }
            if (begin != end) {
                std::size_t idStart = (nJobs - 1) * grainSize;
                jobs.emplace_back([&updateOp, idStart, begin, end]() {
                    auto id = idStart;
                    for (std::random_access_iterator auto it = begin; it != end; ++it, ++id) {
                        updateOp(id, *it);
                    }
                });
            }
        }
    }

    /**
     * Determine a multidimensional cell index based on a position. This function is where the "space centered around
     * origin" assumption enters.
     *
     * @tparam Position Type of position
     * @param pos the position
     * @return multidimensional index
     */
    template<typename Position>
    typename Index::GridDims gridPos(const Position &pos) const {
        typename Index::GridDims projections;
        for (auto i = 0U; i < DIM; ++i) {
            projections[i] = static_cast<typename Index::value_type>(
                    std::max(static_cast<dtype>(0.),
                             static_cast<dtype>(std::floor((pos[i] + .5 * _gridSize[i]) / _cellSize[i]))));
            projections[i] = std::clamp(projections[i], static_cast<typename Index::value_type>(0),
                                        static_cast<typename Index::value_type>(_index[i] - 1));
        }
        return projections;
    }

    /**
     * Returns flat cell index based on position. See gridPos.
     *
     * @tparam Position Type of position.
     * @param pos The position
     * @return flat index pointing to a cell
     */
    template<typename Position>
    std::uint32_t positionToCellIndex(const Position &pos) const {
        return _index.index(gridPos(pos));
    }

    /**
     * The number of cells in this cell linked-list
     */
    auto nCellsTotal() const {
        return _index.size();
    }

    /**
     * Evaluate a functional on particle pairs which contain all particle pairs within the cutoff
     * radius (but potentially more).
     *
     * @tparam F Functional type
     * @param func The function reference, forwarded into worker lambda.
     * @param nJobs Number of jobs.
     */
    template<typename F>
    void forEachParticlePair(F &&func, std::uint32_t nJobs) const {
        // worker function which applies the given function to all neighbors in all cells
        const auto worker = [this, func = std::forward<F>(func)](auto begin, auto end) {
            for (auto i = begin; i != end; ++i) {
                forEachNeighborInCell(func, i);
            }
        };
        // the worker threads, joined upon end of scope
        std::vector<std::jthread> workers;

        const auto grainSize = nCellsTotal() / nJobs;
        std::size_t grainStep;
        for (grainStep = 0; grainStep < nJobs - 1; ++grainStep) {
            workers.emplace_back([&worker, begin = grainStep * grainSize, end = (grainStep + 1) * grainSize]() {
                worker(begin, end);
            });
        }
        if (grainStep * grainSize != nCellsTotal()) {
            workers.emplace_back([&worker, begin = grainStep * grainSize, end = nCellsTotal()]() {
                worker(begin, end);
            });
        }
    }

    /**
     * Applies a user-provided function to all particle pairs in and around a specific cell. Not parallelized.
     *
     * @tparam F The function type
     * @param func The function universal reference
     * @param cellIndex flat index of the cell
     */
    template<typename F>
    void forEachNeighborInCell(F &&func, typename Index::value_type cellIndex) const {
        auto particleId = (*head.at(cellIndex)).load();
        while (particleId != 0) {
            const auto begin = _adjacency.cellsBegin(cellIndex);
            const auto end = _adjacency.cellsEnd(cellIndex);
            for (auto k = begin; k != end; ++k) {
                const auto neighborCellId = *k;
                auto neighborId = (*head.at(neighborCellId)).load();
                while (neighborId != 0) {
                    if (neighborId != particleId) {
                        func(particleId - 1, neighborId - 1);
                    }
                    neighborId = list.at(neighborId);
                }
            }

            particleId = list.at(particleId);
        }
    }

    /**
     * Yields the extent of space.
     */
    const std::array<dtype, DIM> &gridSize() const {
        return _gridSize;
    }

private:
    std::array<dtype, DIM> _cellSize{};
    std::array<dtype, DIM> _gridSize{};
    std::vector<CopyableAtomic<std::size_t>> head{};
    std::vector<std::size_t> list{};
    Index _index{};
    CellAdjacency<Index, periodic> _adjacency{};
    std::unordered_set<std::size_t> types{};
};

}

