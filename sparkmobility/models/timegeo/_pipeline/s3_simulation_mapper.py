import concurrent.futures
import logging
import math
import os
import random
import time
from math import asin, cos, radians, sin, sqrt
from multiprocessing import cpu_count

import numpy as np

logger = logging.getLogger(__name__)

# Global variables to store parameters for the wrapper function
_simulation_params = {}


def _simulate_wrapper(file_index):
    """
    Module-level wrapper function for multiprocessing. Can be pickled.
    """
    # print(f"[DEBUG] _simulate_wrapper started for file_index={file_index}, pid={os.getpid()}")
    params = _simulation_params
    result = simulate(
        file_index=file_index,
        other_locations_file=params.get("other_locations_file"),
        activeness_file=params.get("activeness_file"),
        num_days=params.get("num_days", 1),
        start_slot=params.get("start_slot", 0),
        users_locations_dir=params.get("users_locations_dir"),
        users_parameters_dir=params.get("users_parameters_dir"),
        output_dir=params.get("output_dir"),
        **params.get("kwargs", {}),
    )
    # print(f"[DEBUG] _simulate_wrapper finished for file_index={file_index}, pid={os.getpid()}")
    return result


def simulate(
    file_index,
    other_locations_file="./results/Simulation/otherlocation.txt",
    activeness_file="./results/Simulation/activeness.txt",
    num_days=1,
    start_slot=0,
    users_locations_dir="./results/Simulation/Locations",
    users_parameters_dir="./results/Simulation/Parameters",
    output_dir="./results/Simulation/Mapped",
    break_mean=3.75106,
    break_sd=0.681418,
    break_x0=0.0,
    break_gamma=0.1,
    rho=0.6,
    gamma_val=-0.21,
    min_dist=0.6,
    exponent=-0.86,
):
    # print(f"[DEBUG] simulate started for file_index={file_index}, pid={os.getpid()}")
    # map external parameters to original variable names
    OTHER_LOCATIONS = other_locations_file
    ACTIVENESS_LOCATION = activeness_file
    NUM_DAYS = num_days
    START_SLOT = start_slot
    END_SLOT = 144 * NUM_DAYS - 1 + START_SLOT
    BREAK_MEAN = break_mean
    BREAK_SD = break_sd
    BREAK_X0 = break_x0
    BREAK_GAMMA = break_gamma
    RHO = rho
    GAMMA = gamma_val
    MIN_DIST = min_dist
    EXPONENT = exponent

    usersProcessed = 0

    def genRankSelectionCDF(n):
        rank_select_pdf = np.array(range(1, n + 1))
        rank_select_pdf = list(map(customPow, rank_select_pdf))
        pdf_sum = sum(rank_select_pdf)
        rank_select_pdf = [x / pdf_sum for x in rank_select_pdf]
        rank_select_cdf = np.cumsum(rank_select_pdf)
        return rank_select_cdf

    def loadLocPool():
        f = open(OTHER_LOCATIONS, "r")  # Each line is of form latitude, longitude
        loc_pool = [
            [float(x.split(" ")[1]), float(x.split(" ")[0])] for x in f.readlines()
        ]
        return loc_pool

    def getActiveness(location):
        with open(location, "r") as f:
            pt = []
            for line in f:
                line = line.strip().split(" ")
                pt.append([float(x) for x in line])
        return pt

    def cauchy(loc, scale):
        # Start with a uniform random sample from the open interval (0, 1).
        # But random() returns a sample from the half-open interval [0, 1).
        # In the unlikely event that random() returns 0, try again.
        p = 0.0
        i = 0
        while p == 0.0:
            i += 1
            if i > 10:
                logger.info("Cauchy")
            p = random.random()
        return loc + scale * math.tan(math.pi * (p - 0.5))

    def customPow(x):
        return pow(x, EXPONENT)

    def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(dlon * 0.5) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers. Use 3956 for miles
        return c * r

    RANK_SELECT_CDF = genRankSelectionCDF(100000)
    LOC_POOL = loadLocPool()
    ACTIVENESS_NC, ACTIVENESS_C = getActiveness(ACTIVENESS_LOCATION)
    MAX_CIRCADIAN_C = max(ACTIVENESS_C)
    MAX_CIRCADIAN_NC = max(ACTIVENESS_NC)

    ################################################
    ###########     CORE SIMULATION      ###########
    ################################################

    def findNextLoc(locs, curr_loc, olon, olat):
        numLocs = len(locs)
        r_return = 0.6 * (1 - RHO * pow(numLocs, GAMMA))  # Unconditioned
        p_return = r_return / (r_return + RHO * pow(numLocs, GAMMA))
        if random.random() < p_return:
            visit_count_sum = 0.0
            for j in range(1, numLocs):
                if curr_loc != j:
                    visit_count_sum += locs[j][1]
            visit_cdf = [0] * numLocs
            for j in range(1, numLocs):
                if curr_loc != j:
                    visit_cdf[j] = visit_cdf[j - 1] + locs[j][1] / visit_count_sum
                else:
                    visit_cdf[j] = visit_cdf[j - 1]
            loop_count = 0
            while True:
                loop_count += 1
                if loop_count > 10000:
                    logger.info(
                        f"[WARN] Possible infinite loop in findNextLoc (visited locations) for user {userId}"
                    )
                    # Fall back to returning a random valid location
                    valid_locs = [j for j in range(numLocs) if j != curr_loc]
                    if valid_locs:
                        chosen_loc = random.choice(valid_locs)
                        locs[chosen_loc][1] += 1
                        return chosen_loc
                    else:
                        logger.error(
                            f"[ERROR] No valid locations found for user {userId}"
                        )
                        return 0  # Return home as fallback
                rand_num = random.random()
                for j in range(numLocs):
                    if rand_num < visit_cdf[j] and j != curr_loc:
                        locs[j][1] += 1
                        return j
        else:
            n = len(LOC_POOL)
            dists = [
                [haversine(LOC_POOL[i][0], LOC_POOL[i][1], olon, olat), i]
                for i in range(n)
            ]
            dists.sort()
            begin_index = 0
            for i in range(n):
                if dists[i][0] > MIN_DIST:
                    begin_index = i
                    break
            loop_count = 0
            while True:
                loop_count += 1
                if loop_count > 10000:
                    logger.info(
                        f"[WARN] Possible infinite loop in findNextLoc (new locations) for user {userId}"
                    )
                    # Fall back to using a random location beyond MIN_DIST
                    valid_indices = list(range(begin_index, n))
                    if valid_indices:
                        chosen_idx = random.choice(valid_indices)
                        locs.append(
                            [
                                locs[-1][0] + 1,
                                1,
                                LOC_POOL[dists[chosen_idx][1]][0],
                                LOC_POOL[dists[chosen_idx][1]][1],
                            ]
                        )
                        return locs[-1][0]
                    else:
                        logger.error(
                            f"[ERROR] No valid new locations found beyond MIN_DIST for user {userId}"
                        )
                        return 0  # Return home as fallback
                rand_num = random.random()
                for i in range(begin_index, n):
                    if rand_num < RANK_SELECT_CDF[i - begin_index]:
                        locs.append(
                            [
                                locs[-1][0] + 1,
                                1,
                                LOC_POOL[dists[i][1]][0],
                                LOC_POOL[dists[i][1]][1],
                            ]
                        )
                        return locs[-1][0]

    # Deterine whether a particular slot numbe is inside a list of slots
    # The list of slots consists of slot pairs- (start slot, end slot)
    def inSlots(slot, listOfSlots):
        for rng in listOfSlots:
            if rng[0] <= slot and slot <= rng[1]:
                return True
        return False

    # the input information locs is: locid, num times visited, lon, lat
    # parameters needed: b1, b2, start slot, end slot
    # home is locid 0, 'other' location ids start at 1
    def processUser(locs, b1, b2, nw, workingSlots, workLocation, userId=None):
        # if userId:
        # print(f"[DEBUG] processUser started for file_index={file_index}, user={userId}")
        slot_num = END_SLOT - START_SLOT + 1
        # Initialize location array for each slot
        person_loc = [0] * slot_num
        at_home = 1
        at_work = 0
        curr_loc = 0  # Starting at home
        isWorker = bool(workLocation)

        # MODIFIED- Curated for 1 day
        workingSlotsProcessed = 0
        for i in range(slot_num):
            curr_slot = START_SLOT + i
            if isWorker:
                activeness = ACTIVENESS_C[curr_slot]
                MAX_CIRCADIAN = MAX_CIRCADIAN_C
            else:
                activeness = ACTIVENESS_NC[curr_slot]
                MAX_CIRCADIAN = MAX_CIRCADIAN_NC

            # The processing of time slots at which person is at work
            # is done beforehand

            # A person can be at H, W or O location before the current slot
            # If the currrent slot is a working slot, set/keep person at work
            # If user is at home, he/she can either stay home or go to an
            # 'other' location (not work since we have already modeled work)
            # If user is at 'other' location, he/she can stay there, go home
            # or go to another 'other' location
            if inSlots(curr_slot, workingSlots):
                # if userId:
                # print(f"[DEBUG] User {userId} is in work slot {curr_slot}, work slots: {workingSlots}")
                at_work = 1
                at_home = 0
                curr_loc = "work"
                person_loc[i] = "work"
            elif at_work == 1:
                # if userId:
                #     print(f"[DEBUG] User {userId} leaving work at slot {curr_slot}")
                at_work = 0
                curr_loc = "work"
                workingSlotsProcessed += 1

            # Move daily_slot calculation here, after work slot processing
            daily_slot = curr_slot % 144

            if at_home:
                # print "from home"
                pt = nw * activeness
                # from home to other
                if random.random() < pt:
                    # print "home to other"
                    at_home = 0
                    # decide where to go
                    person_loc[i] = findNextLoc(
                        locs, curr_loc, locs[curr_loc][2], locs[curr_loc][3]
                    )
                    curr_loc = person_loc[i]
                else:
                    # print "stay at home"
                    # keep at home, record it
                    person_loc[i] = 0
                    curr_loc = 0
            elif not at_work:
                # print "from other"
                p_other_move = b1 * nw * activeness
                if curr_loc == "work":
                    p_other_move = 1.1
                if random.random() < p_other_move:
                    # move to home or another other
                    p_other_home = 1 - b2 * nw * activeness
                    p_home_circadian = 1 - activeness / MAX_CIRCADIAN
                    if daily_slot > 96:  # Start after 4pm
                        p_other_home = max(p_other_home, p_home_circadian)

                    # Modification- Between consecutive work slots
                    if len(workingSlots) == 2 and workingSlotsProcessed == 1:
                        p_other_home = -1
                    if random.random() < p_other_home:
                        # print "other to home"
                        at_home = 1
                        person_loc[i] = 0
                        curr_loc = 0
                    else:
                        # other to other or work to other
                        # decide where to go
                        if curr_loc == "work":
                            person_loc[i] = findNextLoc(
                                locs, curr_loc, workLocation[0], workLocation[1]
                            )
                        else:
                            person_loc[i] = findNextLoc(
                                locs, curr_loc, locs[curr_loc][2], locs[curr_loc][3]
                            )
                        curr_loc = person_loc[i]
                else:
                    # else keep at the current other place
                    person_loc[i] = curr_loc
        # if userId:
        #     print(f"[DEBUG] processUser finished for file_index={file_index}, user={userId}")
        return locs, person_loc

    ################################################
    ########### END OF CORE SIMULATION   ###########
    ################################################

    # The job of the remaining functions is to feed data about a
    # particular user to the 'processUser' function.
    #
    # The processUser function takes in the following arguments-
    # 1. locs           :       Details of locations previously visited by user
    #                           [location id, num times visited, lon, lat]
    # 2. b1             :       Parameter Beta1
    # 3. b2             :       Parameter Beta2
    # 4. nw             :       Parameter NW
    # 5. workingSlots   :       Interval in which user is working. Of form
    #                           [[start1, stop1], [start2, stop2]...]
    # 6. workLocation   :       Work location of user [longitude, latitude]

    os.makedirs(output_dir, exist_ok=True)
    numErrors = 0
    out_file = os.path.join(output_dir, f"simulationResults_{file_index}.txt")
    with open(out_file, "w") as g:
        userLocations = {}
        userWorkLocations = {}
        loc_file = os.path.join(users_locations_dir, f"usersLocations_{file_index}.txt")
        with open(loc_file, "r") as f:
            for line in f:
                line = line.strip().split(" ")
                if len(line) == 1:
                    currentUser = line[0]
                    userLocations[currentUser] = []
                    otherLocationIndex = 1
                elif currentUser:
                    line[1] = int(line[1])
                    line[2] = float(line[2])
                    line[3] = float(line[3])
                    if line[0] == "h":
                        line[0] = 0
                        userLocations[currentUser].append(line)
                    elif line[0] == "w":
                        userWorkLocations[currentUser] = [line[2], line[3]]
                    else:
                        line[0] = otherLocationIndex
                        otherLocationIndex += 1
                        userLocations[currentUser].append(line)

        param_file = os.path.join(
            users_parameters_dir, f"usersParameters_{file_index}.txt"
        )
        with open(param_file, "r") as f:
            for line in f:
                line = line.strip().split(" ")
                userId = line[0]  # The user ID is actually the first column
                workingStatus = int(
                    float(line[5])
                )  # Working status is at index 5, handle float values

                # MODIFICATION
                g.write(userId + "\n")
                if workingStatus:
                    # Parse work slots - they are individual slot numbers starting from field 9
                    work_slots_raw = line[8:] if len(line) > 8 else []
                    workingSlots = []

                    # Convert individual slot numbers to time ranges
                    if work_slots_raw:
                        slots = [int(x) for x in work_slots_raw if x.strip()]
                        if slots:
                            # Group consecutive slots into ranges
                            ranges = []
                            start = slots[0]
                            end = slots[0]

                            for slot in slots[1:]:
                                if slot == end + 1:
                                    end = slot
                                else:
                                    ranges.append([start, end])
                                    start = slot
                                    end = slot

                            ranges.append([start, end])
                            workingSlots = ranges
                            # print(f"[DEBUG] User {userId} has {len(workingSlots)} work slot ranges: {workingSlots}")

                    workLocation = userWorkLocations.get(userId, [])
                else:
                    workingSlots = []
                    workLocation = []
                b1, b2, nw = float(line[1]), float(line[2]), float(line[3])
                locsVisited = userLocations.get(userId, [])

                modifiedWorkingSlots = []
                for rng in workingSlots:
                    take_lunch = random.random()
                    wk_dur_m = (rng[1] - rng[0]) * 10
                    if take_lunch < 0.22194 and wk_dur_m > 0:
                        choosen = 0
                        counter = 0
                        while choosen == 0:
                            counter += 1
                            if counter > 10000:
                                logger.info("BreakCounter")
                                break
                            bk_dur_m = np.random.lognormal(BREAK_MEAN, BREAK_SD, 1)[0]
                            if bk_dur_m < wk_dur_m:
                                choosen = 1

                        # Introducing the Cauchy distribution for break timestamp
                        ind = 0
                        tc = 0
                        while ind == 0:
                            tc += 1
                            if tc > 10000:
                                logger.info("BreakTsCounter")
                                break
                            s1 = cauchy(BREAK_X0, BREAK_GAMMA)
                            if -0.5 < s1 < 0.5:
                                ind = 1

                        diff_m = wk_dur_m - bk_dur_m
                        shift_m = diff_m * s1  # in minutes
                        bk_center_m = 10 * (rng[0] + rng[1]) / 2 + shift_m
                        bk_start_m = bk_center_m - bk_dur_m / 2.0
                        bk_end_m = bk_center_m + bk_dur_m / 2.0
                        modifiedWorkingSlots.append([rng[0], int(bk_start_m / 10)])
                        modifiedWorkingSlots.append([int(bk_end_m / 10), rng[1]])
                    else:
                        modifiedWorkingSlots.append(rng)
                try:
                    locs, person_loc = processUser(
                        locsVisited,
                        b1,
                        b2,
                        nw,
                        modifiedWorkingSlots,
                        workLocation,
                        userId,
                    )
                except Exception as e:
                    logger.error(f"error, userId: {userId}, exception: {e}")
                    continue

                try:
                    differentLocs = set(person_loc)
                    mapMatching = {}

                    for dl in differentLocs:
                        if dl == "work":
                            locType = "w"
                            if len(workLocation) >= 2:
                                location = workLocation
                            else:
                                # Fallback: use home location if work location is not available
                                logger.info(
                                    f"[WARN] No work location found for user {userId}, using home location as fallback"
                                )
                                matches = [
                                    x for x in locs if x[0] == 0
                                ]  # Home location
                                if matches:
                                    location = matches[0][2:4]
                                else:
                                    numErrors += 1
                                    continue
                        elif dl == 0:
                            locType = "h"
                        else:
                            locType = "o"

                        if dl != "work":
                            matches = [x for x in locs if x[0] == dl]
                            if not matches:
                                numErrors += 1
                                continue
                            location = matches[0][2:4]

                        mapMatching[dl] = [locType, location[0], location[1]]

                    for slot_dl in person_loc:
                        entry = mapMatching.get(slot_dl)
                        if entry:
                            g.write(" ".join(map(str, entry)) + "\n")

                except Exception as e:
                    logger.info(
                        f"[WARN] Failed to write mapMatching for user {userId}: {e}"
                    )

                usersProcessed += 1

        logger.error(f"Users processed: {usersProcessed}, error: {numErrors}")

    logger.info(f"Process {file_index}: Output written to: {out_file}")
    # print(f"[DEBUG] simulate finished for file_index={file_index}, pid={os.getpid()}")


def simulate_all_parallel_optimized(
    num_cpus=16,
    other_locations_file="./results/Simulation/otherlocation.txt",
    activeness_file="./results/Simulation/activeness.txt",
    num_days=1,
    start_slot=0,
    users_locations_dir="./results/Simulation/Locations",
    users_parameters_dir="./results/Simulation/Parameters",
    output_dir="./results/Simulation/Mapped",
    **kwargs,
):
    """
    Run simulation for all file indices in parallel using optimized multiprocessing.
    Uses concurrent futures for better resource management and progress tracking.
    """
    # print(f"[DEBUG] Starting simulate_all_parallel_optimized with {num_cpus} CPUs")

    # Set global parameters for the wrapper function
    global _simulation_params
    _simulation_params = {
        "other_locations_file": other_locations_file,
        "activeness_file": activeness_file,
        "num_days": num_days,
        "start_slot": start_slot,
        "users_locations_dir": users_locations_dir,
        "users_parameters_dir": users_parameters_dir,
        "output_dir": output_dir,
        "kwargs": kwargs,
    }

    # Check which files actually exist to avoid processing empty files
    active_indices = []
    for i in range(num_cpus):
        locations_file = os.path.join(users_locations_dir, f"usersLocations_{i}.txt")
        parameters_file = os.path.join(users_parameters_dir, f"usersParameters_{i}.txt")

        if (
            os.path.exists(locations_file)
            and os.path.getsize(locations_file) > 0
            and os.path.exists(parameters_file)
            and os.path.getsize(parameters_file) > 0
        ):
            active_indices.append(i)

    # print(f"[DEBUG] Active indices: {active_indices}")
    logger.info(
        f"Found {len(active_indices)} active file pairs out of {num_cpus} total"
    )

    if not active_indices:
        logger.info("No active files found for simulation")
        return

    start_time = time.time()

    # Use ProcessPoolExecutor for better resource management
    max_workers = min(len(active_indices), cpu_count())
    logger.info(f"Using {max_workers} worker processes for {len(active_indices)} tasks")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(_simulate_wrapper, i): i for i in active_indices
        }

        # Process completed tasks and track progress
        completed = 0
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                # print(f"[DEBUG] Waiting for result from file index {index}")
                future.result(timeout=600)  # 10 minute timeout per file
                completed += 1
                elapsed = time.time() - start_time
                progress = (completed / len(active_indices)) * 100
                eta = (
                    (elapsed / completed) * (len(active_indices) - completed)
                    if completed > 0
                    else 0
                )
                logger.info(
                    "completed %d/%d (%.1f%%) file %d elapsed=%.1fs eta=%.1fs",
                    completed,
                    len(active_indices),
                    progress,
                    index,
                    elapsed,
                    eta,
                )
            except concurrent.futures.TimeoutError:
                logger.error(f"[ERROR] Timeout for file index {index}")
            except Exception as exc:
                logger.error(
                    f"[ERROR] Simulation for file {index} generated an exception: {exc}"
                )

    total_time = time.time() - start_time
    logger.info(
        f"Completed optimized simulation for all {len(active_indices)} active processes in {total_time:.1f}s"
    )


def simulate_batch_optimized(
    batch_indices,
    other_locations_file="./results/Simulation/otherlocation.txt",
    activeness_file="./results/Simulation/activeness.txt",
    num_days=1,
    start_slot=0,
    users_locations_dir="./results/Simulation/Locations",
    users_parameters_dir="./results/Simulation/Parameters",
    output_dir="./results/Simulation/Mapped",
    **kwargs,
):
    """
    Process multiple simulation files in a single process to reduce overhead.
    """
    logger.info(f"Processing batch: {batch_indices}")

    for file_index in batch_indices:
        try:
            simulate(
                file_index=file_index,
                other_locations_file=other_locations_file,
                activeness_file=activeness_file,
                num_days=num_days,
                start_slot=start_slot,
                users_locations_dir=users_locations_dir,
                users_parameters_dir=users_parameters_dir,
                output_dir=output_dir,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Error processing file {file_index}: {e}")

    return len(batch_indices)


def simulate_all_parallel_batched(
    num_cpus=16,
    batch_size=4,
    other_locations_file="./results/Simulation/otherlocation.txt",
    activeness_file="./results/Simulation/activeness.txt",
    num_days=1,
    start_slot=0,
    users_locations_dir="./results/Simulation/Locations",
    users_parameters_dir="./results/Simulation/Parameters",
    output_dir="./results/Simulation/Mapped",
    **kwargs,
):
    """
    Run simulation using batched processing to reduce process creation overhead.
    """
    logger.info(
        f"Starting batched parallel simulation with {num_cpus} files, batch size {batch_size}..."
    )

    # Check which files actually exist
    active_indices = []
    for i in range(num_cpus):
        locations_file = os.path.join(users_locations_dir, f"usersLocations_{i}.txt")
        parameters_file = os.path.join(users_parameters_dir, f"usersParameters_{i}.txt")

        if (
            os.path.exists(locations_file)
            and os.path.getsize(locations_file) > 0
            and os.path.exists(parameters_file)
            and os.path.getsize(parameters_file) > 0
        ):
            active_indices.append(i)

    if not active_indices:
        logger.info("No active files found for simulation")
        return

    # Create batches
    batches = []
    for i in range(0, len(active_indices), batch_size):
        batch = active_indices[i : i + batch_size]
        batches.append(batch)

    logger.info(
        f"Created {len(batches)} batches from {len(active_indices)} active files"
    )

    start_time = time.time()

    # Process batches in parallel
    max_workers = min(len(batches), cpu_count())
    logger.info(f"Using {max_workers} worker processes for {len(batches)} batches")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit batch tasks
        futures = []
        for batch in batches:
            future = executor.submit(
                simulate_batch_optimized,
                batch,
                other_locations_file=other_locations_file,
                activeness_file=activeness_file,
                num_days=num_days,
                start_slot=start_slot,
                users_locations_dir=users_locations_dir,
                users_parameters_dir=users_parameters_dir,
                output_dir=output_dir,
                **kwargs,
            )
            futures.append(future)

        # Track progress
        completed = 0
        total_files_processed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                files_in_batch = future.result()
                total_files_processed += files_in_batch
                completed += 1
                elapsed = time.time() - start_time
                progress = (completed / len(batches)) * 100
                eta = (
                    (elapsed / completed) * (len(batches) - completed)
                    if completed > 0
                    else 0
                )
                logger.info(
                    "completed batch %d/%d (%.1f%%) files=%d elapsed=%.1fs eta=%.1fs",
                    completed,
                    len(batches),
                    progress,
                    total_files_processed,
                    elapsed,
                    eta,
                )
            except Exception as exc:
                logger.exception("batch generated an exception: %s", exc)

    total_time = time.time() - start_time
    logger.info(
        f"Completed batched simulation for {total_files_processed} files in {total_time:.1f}s"
    )


def simulate_sequential(
    num_cpus=16,
    other_locations_file="./results/Simulation/otherlocation.txt",
    activeness_file="./results/Simulation/activeness.txt",
    num_days=1,
    start_slot=0,
    users_locations_dir="./results/Simulation/Locations",
    users_parameters_dir="./results/Simulation/Parameters",
    output_dir="./results/Simulation/Mapped",
    **kwargs,
):
    """
    Run simulation for all file indices sequentially (for debugging).
    """
    for file_index in range(num_cpus):
        simulate(
            file_index=file_index,
            other_locations_file=other_locations_file,
            activeness_file=activeness_file,
            num_days=num_days,
            start_slot=start_slot,
            users_locations_dir=users_locations_dir,
            users_parameters_dir=users_parameters_dir,
            output_dir=output_dir,
            **kwargs,
        )

    logger.info(f"Completed sequential simulation for all {num_cpus} processes")


# Keep original function for backward compatibility
def simulate_all_parallel(
    num_cpus=16,
    other_locations_file="./results/Simulation/otherlocation.txt",
    activeness_file="./results/Simulation/activeness.txt",
    num_days=1,
    start_slot=0,
    users_locations_dir="./results/Simulation/Locations",
    users_parameters_dir="./results/Simulation/Parameters",
    output_dir="./results/Simulation/Mapped",
    **kwargs,
):
    """
    Run simulation for all file indices in parallel using multiprocessing.
    """
    # Use optimized version by default
    return simulate_all_parallel_optimized(
        num_cpus=num_cpus,
        other_locations_file=other_locations_file,
        activeness_file=activeness_file,
        num_days=num_days,
        start_slot=start_slot,
        users_locations_dir=users_locations_dir,
        users_parameters_dir=users_parameters_dir,
        output_dir=output_dir,
        **kwargs,
    )
